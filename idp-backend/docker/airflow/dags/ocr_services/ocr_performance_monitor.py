"""
OCR Performance Monitoring
Tracks and reports OCR performance metrics for optimization
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class OCRPerformanceMonitor:
    """Monitor and track OCR performance metrics"""
    
    def __init__(self, log_dir: str = "/opt/airflow/logs/ocr_performance"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.current_session = {
            'session_id': f"ocr_session_{int(time.time())}",
            'start_time': datetime.utcnow().isoformat(),
            'documents_processed': 0,
            'total_pages': 0,
            'total_processing_time': 0,
            'engine_usage': defaultdict(int),
            'fallback_usage': defaultdict(int),
            'error_count': 0,
            'performance_by_engine': defaultdict(list),
            'document_stats': []
        }
    
    def log_document_processing(self, 
                              document_path: str,
                              engine_used: str,
                              pages_processed: int,
                              processing_time: float,
                              confidence_scores: List[float],
                              fallback_used: bool = False,
                              error_occurred: bool = False):
        """
        Log processing metrics for a document
        
        Args:
            document_path: Path to processed document
            engine_used: OCR engine that was used
            pages_processed: Number of pages processed
            processing_time: Total processing time in seconds
            confidence_scores: List of confidence scores per page
            fallback_used: Whether fallback engine was used
            error_occurred: Whether an error occurred during processing
        """
        
        document_name = os.path.basename(document_path)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        doc_stats = {
            'document_name': document_name,
            'document_path': document_path,
            'engine_used': engine_used,
            'pages_processed': pages_processed,
            'processing_time': processing_time,
            'avg_confidence': avg_confidence,
            'confidence_scores': confidence_scores,
            'fallback_used': fallback_used,
            'error_occurred': error_occurred,
            'timestamp': datetime.utcnow().isoformat(),
            'pages_per_second': pages_processed / processing_time if processing_time > 0 else 0
        }
        
        # Update session stats
        self.current_session['documents_processed'] += 1
        self.current_session['total_pages'] += pages_processed
        self.current_session['total_processing_time'] += processing_time
        self.current_session['engine_usage'][engine_used] += 1
        
        if fallback_used:
            self.current_session['fallback_usage'][engine_used] += 1
        
        if error_occurred:
            self.current_session['error_count'] += 1
        
        # Track performance by engine
        self.current_session['performance_by_engine'][engine_used].append({
            'processing_time': processing_time,
            'pages_processed': pages_processed,
            'avg_confidence': avg_confidence,
            'pages_per_second': doc_stats['pages_per_second']
        })
        
        self.current_session['document_stats'].append(doc_stats)
        
        # Log to file periodically
        if self.current_session['documents_processed'] % 10 == 0:
            self._save_session_log()
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session performance"""
        
        session = self.current_session
        
        # Calculate averages
        avg_processing_time = (session['total_processing_time'] / session['documents_processed'] 
                             if session['documents_processed'] > 0 else 0)
        
        avg_pages_per_doc = (session['total_pages'] / session['documents_processed'] 
                           if session['documents_processed'] > 0 else 0)
        
        overall_pages_per_second = (session['total_pages'] / session['total_processing_time'] 
                                  if session['total_processing_time'] > 0 else 0)
        
        # Calculate engine performance
        engine_performance = {}
        for engine, performances in session['performance_by_engine'].items():
            if performances:
                engine_performance[engine] = {
                    'avg_processing_time': sum(p['processing_time'] for p in performances) / len(performances),
                    'avg_confidence': sum(p['avg_confidence'] for p in performances) / len(performances),
                    'avg_pages_per_second': sum(p['pages_per_second'] for p in performances) / len(performances),
                    'documents_processed': len(performances)
                }
        
        # Calculate fallback rates
        fallback_rates = {}
        for engine, fallback_count in session['fallback_usage'].items():
            total_usage = session['engine_usage'][engine]
            fallback_rates[engine] = (fallback_count / total_usage * 100) if total_usage > 0 else 0
        
        return {
            'session_id': session['session_id'],
            'start_time': session['start_time'],
            'current_time': datetime.utcnow().isoformat(),
            'documents_processed': session['documents_processed'],
            'total_pages': session['total_pages'],
            'total_processing_time': session['total_processing_time'],
            'avg_processing_time_per_doc': avg_processing_time,
            'avg_pages_per_doc': avg_pages_per_doc,
            'overall_pages_per_second': overall_pages_per_second,
            'engine_usage': dict(session['engine_usage']),
            'engine_performance': engine_performance,
            'fallback_rates': fallback_rates,
            'error_count': session['error_count'],
            'error_rate': (session['error_count'] / session['documents_processed'] * 100) 
                         if session['documents_processed'] > 0 else 0
        }
    
    def get_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        
        summary = self.get_session_summary()
        recommendations = []
        
        # Check overall performance
        if summary['overall_pages_per_second'] < 0.5:
            recommendations.append(
                "Overall processing speed is slow (<0.5 pages/sec). "
                "Consider reducing DPI or enabling parallel processing."
            )
        
        # Check engine performance
        engine_perf = summary['engine_performance']
        if len(engine_perf) > 1:
            # Compare engine performance
            fastest_engine = max(engine_perf.keys(), 
                               key=lambda e: engine_perf[e]['avg_pages_per_second'])
            slowest_engine = min(engine_perf.keys(), 
                               key=lambda e: engine_perf[e]['avg_pages_per_second'])
            
            fastest_speed = engine_perf[fastest_engine]['avg_pages_per_second']
            slowest_speed = engine_perf[slowest_engine]['avg_pages_per_second']
            
            if fastest_speed > slowest_speed * 2:
                recommendations.append(
                    f"Engine '{fastest_engine}' is significantly faster than '{slowest_engine}'. "
                    f"Consider using '{fastest_engine}' as primary engine."
                )
        
        # Check fallback rates
        for engine, rate in summary['fallback_rates'].items():
            if rate > 30:
                recommendations.append(
                    f"High fallback rate for {engine} ({rate:.1f}%). "
                    f"Consider adjusting OCR parameters or switching primary engine."
                )
        
        # Check error rates
        if summary['error_rate'] > 10:
            recommendations.append(
                f"High error rate ({summary['error_rate']:.1f}%). "
                f"Check document quality and OCR configuration."
            )
        
        # Check confidence scores
        for engine, perf in engine_perf.items():
            if perf['avg_confidence'] < 70:
                recommendations.append(
                    f"Low confidence scores for {engine} ({perf['avg_confidence']:.1f}%). "
                    f"Consider increasing DPI or improving image preprocessing."
                )
        
        return recommendations
    
    def _save_session_log(self):
        """Save current session to log file"""
        
        log_file = os.path.join(
            self.log_dir, 
            f"ocr_performance_{datetime.utcnow().strftime('%Y%m%d')}.json"
        )
        
        try:
            # Load existing logs
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = {'sessions': []}
            
            # Update or add current session
            session_exists = False
            for i, session in enumerate(logs['sessions']):
                if session['session_id'] == self.current_session['session_id']:
                    logs['sessions'][i] = self.current_session.copy()
                    session_exists = True
                    break
            
            if not session_exists:
                logs['sessions'].append(self.current_session.copy())
            
            # Save updated logs
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save performance log: {e}")
    
    def finalize_session(self):
        """Finalize and save the current session"""
        
        self.current_session['end_time'] = datetime.utcnow().isoformat()
        self._save_session_log()
        
        # Print summary
        summary = self.get_session_summary()
        recommendations = self.get_performance_recommendations()
        
        logger.info("OCR Performance Session Summary:")
        logger.info(f"Documents processed: {summary['documents_processed']}")
        logger.info(f"Total pages: {summary['total_pages']}")
        logger.info(f"Processing speed: {summary['overall_pages_per_second']:.2f} pages/sec")
        logger.info(f"Error rate: {summary['error_rate']:.1f}%")
        
        if recommendations:
            logger.info("Performance Recommendations:")
            for rec in recommendations:
                logger.info(f"- {rec}")
    
    @staticmethod
    def load_historical_performance(log_dir: str = "/opt/airflow/logs/ocr_performance", 
                                  days: int = 7) -> Dict:
        """
        Load historical performance data
        
        Args:
            log_dir: Directory containing performance logs
            days: Number of days to look back
        
        Returns:
            Historical performance data
        """
        
        historical_data = {
            'sessions': [],
            'summary': {
                'total_documents': 0,
                'total_pages': 0,
                'avg_pages_per_second': 0,
                'engine_usage': defaultdict(int),
                'avg_error_rate': 0
            }
        }
        
        try:
            # Get log files from the last N days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            for i in range(days):
                date = start_date + timedelta(days=i)
                log_file = os.path.join(
                    log_dir, 
                    f"ocr_performance_{date.strftime('%Y%m%d')}.json"
                )
                
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        daily_logs = json.load(f)
                        historical_data['sessions'].extend(daily_logs.get('sessions', []))
            
            # Calculate summary statistics
            if historical_data['sessions']:
                total_docs = sum(s['documents_processed'] for s in historical_data['sessions'])
                total_pages = sum(s['total_pages'] for s in historical_data['sessions'])
                total_time = sum(s['total_processing_time'] for s in historical_data['sessions'])
                total_errors = sum(s['error_count'] for s in historical_data['sessions'])
                
                historical_data['summary'] = {
                    'total_documents': total_docs,
                    'total_pages': total_pages,
                    'avg_pages_per_second': total_pages / total_time if total_time > 0 else 0,
                    'avg_error_rate': (total_errors / total_docs * 100) if total_docs > 0 else 0,
                    'sessions_count': len(historical_data['sessions'])
                }
                
                # Aggregate engine usage
                for session in historical_data['sessions']:
                    for engine, count in session['engine_usage'].items():
                        historical_data['summary']['engine_usage'][engine] += count
        
        except Exception as e:
            logger.error(f"Failed to load historical performance data: {e}")
        
        return historical_data


# Global performance monitor instance
performance_monitor = OCRPerformanceMonitor()


def log_ocr_performance(document_path: str, 
                       engine_used: str,
                       pages_processed: int,
                       processing_time: float,
                       confidence_scores: List[float],
                       fallback_used: bool = False,
                       error_occurred: bool = False):
    """
    Convenience function to log OCR performance
    
    Args:
        document_path: Path to processed document
        engine_used: OCR engine that was used
        pages_processed: Number of pages processed
        processing_time: Total processing time in seconds
        confidence_scores: List of confidence scores per page
        fallback_used: Whether fallback engine was used
        error_occurred: Whether an error occurred during processing
    """
    performance_monitor.log_document_processing(
        document_path=document_path,
        engine_used=engine_used,
        pages_processed=pages_processed,
        processing_time=processing_time,
        confidence_scores=confidence_scores,
        fallback_used=fallback_used,
        error_occurred=error_occurred
    )


def get_performance_summary() -> Dict:
    """Get current session performance summary"""
    return performance_monitor.get_session_summary()


def get_performance_recommendations() -> List[str]:
    """Get performance optimization recommendations"""
    return performance_monitor.get_performance_recommendations()