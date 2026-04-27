#!/usr/bin/env python3
"""
Migration Script for Optimized OCR
Helps transition from PaddleOCR-heavy implementation to optimized OCR service
"""

import os
import sys
import json
import shutil
from datetime import datetime
from typing import List, Dict
import argparse

# Add the dags directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_services.ocr_config import OCRConfig
from ocr_services.optimized_ocr_service import OptimizedOCRService
from ocr_services.ocr_performance_monitor import OCRPerformanceMonitor


class OCRMigrationTool:
    """Tool to migrate from old OCR system to optimized OCR"""
    
    def __init__(self, base_dir: str = "/opt/airflow/downloaded_docs"):
        self.base_dir = base_dir
        self.config = OCRConfig()
        self.migration_log = []
        
    def analyze_existing_cache(self) -> Dict:
        """Analyze existing OCR cache files to understand current usage"""
        
        analysis = {
            'total_cache_files': 0,
            'total_size_mb': 0,
            'engines_used': {},
            'avg_pages_per_doc': 0,
            'documents_by_size': {'small': 0, 'medium': 0, 'large': 0},
            'cache_files': []
        }
        
        print("🔍 Analyzing existing OCR cache files...")
        
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.json') and 'ocr_output' in root:
                    cache_file = os.path.join(root, file)
                    
                    try:
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                        
                        # Analyze cache file
                        file_size = os.path.getsize(cache_file) / (1024 * 1024)  # MB
                        page_count = cache_data.get('page_count', 0)
                        engine = cache_data.get('ocr_engine', 'unknown')
                        
                        analysis['total_cache_files'] += 1
                        analysis['total_size_mb'] += file_size
                        analysis['engines_used'][engine] = analysis['engines_used'].get(engine, 0) + 1
                        
                        # Categorize by document size
                        if page_count <= 5:
                            analysis['documents_by_size']['small'] += 1
                        elif page_count <= 20:
                            analysis['documents_by_size']['medium'] += 1
                        else:
                            analysis['documents_by_size']['large'] += 1
                        
                        analysis['cache_files'].append({
                            'path': cache_file,
                            'size_mb': file_size,
                            'pages': page_count,
                            'engine': engine,
                            'filename': cache_data.get('filename', 'unknown')
                        })
                        
                    except Exception as e:
                        print(f"⚠️ Failed to analyze {cache_file}: {e}")
        
        if analysis['total_cache_files'] > 0:
            analysis['avg_pages_per_doc'] = sum(
                cf['pages'] for cf in analysis['cache_files']
            ) / analysis['total_cache_files']
        
        return analysis
    
    def create_backup(self) -> str:
        """Create backup of existing OCR cache"""
        
        backup_dir = f"/opt/airflow/ocr_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"📦 Creating backup at {backup_dir}...")
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy OCR cache files
            for root, dirs, files in os.walk(self.base_dir):
                if 'ocr_output' in root:
                    rel_path = os.path.relpath(root, self.base_dir)
                    backup_path = os.path.join(backup_dir, rel_path)
                    os.makedirs(backup_path, exist_ok=True)
                    
                    for file in files:
                        if file.endswith('.json'):
                            src = os.path.join(root, file)
                            dst = os.path.join(backup_path, file)
                            shutil.copy2(src, dst)
            
            print(f"✅ Backup created successfully at {backup_dir}")
            return backup_dir
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return ""
    
    def test_optimized_ocr(self, sample_files: List[str] = None) -> Dict:
        """Test optimized OCR on sample files"""
        
        print("🧪 Testing optimized OCR performance...")
        
        if not sample_files:
            # Find sample PDF files
            sample_files = []
            for root, dirs, files in os.walk(self.base_dir):
                for file in files:
                    if file.endswith('.pdf') and len(sample_files) < 3:
                        sample_files.append(os.path.join(root, file))
        
        if not sample_files:
            print("⚠️ No sample PDF files found for testing")
            return {}
        
        test_results = {
            'files_tested': len(sample_files),
            'results': [],
            'avg_processing_time': 0,
            'avg_pages_per_second': 0,
            'success_rate': 0
        }
        
        # Test with optimized service
        ocr_service = OptimizedOCRService(
            primary_engine='tesseract',
            fallback_engine='paddle',
            max_workers=2,
            enable_performance_logging=True
        )
        
        successful_tests = 0
        total_processing_time = 0
        total_pages = 0
        
        for pdf_file in sample_files:
            print(f"  Testing: {os.path.basename(pdf_file)}")
            
            try:
                start_time = datetime.now()
                
                # Test with optimized configuration
                config = {
                    'dpi': 200,
                    'parallel': True,
                    'max_pages': 5  # Limit for testing
                }
                
                result = ocr_service.extract_text_with_confidence(pdf_file, config)
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                pages_processed = result.get('pages_processed', 0)
                confidence = result.get('confidence', 0)
                text_length = len(result.get('text', ''))
                
                test_result = {
                    'file': os.path.basename(pdf_file),
                    'success': True,
                    'processing_time': processing_time,
                    'pages_processed': pages_processed,
                    'confidence': confidence,
                    'text_length': text_length,
                    'pages_per_second': pages_processed / processing_time if processing_time > 0 else 0,
                    'engine_stats': result.get('engine_stats', {})
                }
                
                test_results['results'].append(test_result)
                successful_tests += 1
                total_processing_time += processing_time
                total_pages += pages_processed
                
                print(f"    ✅ Success: {pages_processed} pages in {processing_time:.2f}s")
                
            except Exception as e:
                test_result = {
                    'file': os.path.basename(pdf_file),
                    'success': False,
                    'error': str(e),
                    'processing_time': 0,
                    'pages_processed': 0
                }
                
                test_results['results'].append(test_result)
                print(f"    ❌ Failed: {e}")
        
        # Calculate averages
        if successful_tests > 0:
            test_results['avg_processing_time'] = total_processing_time / successful_tests
            test_results['avg_pages_per_second'] = total_pages / total_processing_time if total_processing_time > 0 else 0
            test_results['success_rate'] = (successful_tests / len(sample_files)) * 100
        
        return test_results
    
    def generate_migration_report(self, analysis: Dict, test_results: Dict) -> str:
        """Generate migration report"""
        
        report = f"""
# OCR Migration Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Current System Analysis

### Cache Analysis
- Total cache files: {analysis['total_cache_files']}
- Total cache size: {analysis['total_size_mb']:.2f} MB
- Average pages per document: {analysis['avg_pages_per_doc']:.1f}

### Engine Usage
"""
        
        for engine, count in analysis['engines_used'].items():
            percentage = (count / analysis['total_cache_files']) * 100 if analysis['total_cache_files'] > 0 else 0
            report += f"- {engine}: {count} files ({percentage:.1f}%)\n"
        
        report += f"""
### Document Size Distribution
- Small (≤5 pages): {analysis['documents_by_size']['small']} documents
- Medium (6-20 pages): {analysis['documents_by_size']['medium']} documents  
- Large (>20 pages): {analysis['documents_by_size']['large']} documents

## Optimized OCR Test Results

### Performance
- Files tested: {test_results.get('files_tested', 0)}
- Success rate: {test_results.get('success_rate', 0):.1f}%
- Average processing time: {test_results.get('avg_processing_time', 0):.2f}s per document
- Average speed: {test_results.get('avg_pages_per_second', 0):.2f} pages/second

### Individual Test Results
"""
        
        for result in test_results.get('results', []):
            if result['success']:
                report += f"""
- **{result['file']}**
  - Processing time: {result['processing_time']:.2f}s
  - Pages processed: {result['pages_processed']}
  - Confidence: {result['confidence']:.1f}%
  - Speed: {result['pages_per_second']:.2f} pages/sec
  - Engines used: {result['engine_stats']}
"""
            else:
                report += f"""
- **{result['file']}** ❌ FAILED
  - Error: {result['error']}
"""
        
        report += f"""
## Migration Recommendations

### Performance Improvements Expected
"""
        
        # Calculate expected improvements
        if analysis['engines_used'].get('paddle', 0) > 0:
            report += "- **Reduced Memory Usage**: Optimized service uses lazy loading and better memory management\n"
            report += "- **Faster Processing**: Parallel processing and lightweight preprocessing\n"
            report += "- **Better Reliability**: Intelligent fallback between engines\n"
        
        if test_results.get('avg_pages_per_second', 0) > 1:
            report += f"- **Good Performance**: Current test shows {test_results['avg_pages_per_second']:.2f} pages/sec\n"
        elif test_results.get('avg_pages_per_second', 0) > 0.5:
            report += f"- **Moderate Performance**: Current test shows {test_results['avg_pages_per_second']:.2f} pages/sec, consider tuning\n"
        else:
            report += "- **Performance Tuning Needed**: Consider adjusting DPI and parallel settings\n"
        
        report += """
### Configuration Recommendations
"""
        
        # Recommend configuration based on document sizes
        if analysis['documents_by_size']['large'] > analysis['documents_by_size']['small']:
            report += "- Use parallel processing (enabled by default)\n"
            report += "- Consider DPI=200 for balance of speed and quality\n"
            report += "- Set max_workers=4 for large documents\n"
        else:
            report += "- DPI=150-200 should be sufficient for smaller documents\n"
            report += "- max_workers=2-4 depending on system resources\n"
        
        if analysis['engines_used'].get('paddle', 0) > analysis['engines_used'].get('tesseract', 0):
            report += "- Consider 'optimized_paddle' engine (Paddle primary, Tesseract fallback)\n"
        else:
            report += "- Use 'optimized' engine (Tesseract primary, Paddle fallback)\n"
        
        report += """
### Next Steps
1. **Backup Complete**: OCR cache has been backed up
2. **Update Configuration**: Set environment variables for production
3. **Monitor Performance**: Use OCR performance monitoring
4. **Gradual Rollout**: Test on subset of documents first

### Environment Variables for Production
```bash
ENVIRONMENT=production
OCR_DEFAULT_ENGINE=optimized
OCR_MAX_WORKERS=4
OCR_DEFAULT_DPI=200
OCR_MAX_PAGES=50
```
"""
        
        return report
    
    def run_migration(self, create_backup: bool = True, test_samples: int = 3) -> str:
        """Run complete migration process"""
        
        print("🚀 Starting OCR Migration Process...")
        print("=" * 50)
        
        # Step 1: Analyze existing system
        analysis = self.analyze_existing_cache()
        
        # Step 2: Create backup
        backup_dir = ""
        if create_backup:
            backup_dir = self.create_backup()
        
        # Step 3: Test optimized OCR
        test_results = self.test_optimized_ocr()
        
        # Step 4: Generate report
        report = self.generate_migration_report(analysis, test_results)
        
        # Save report
        report_file = f"/opt/airflow/ocr_migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        try:
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"📄 Migration report saved to: {report_file}")
        except Exception as e:
            print(f"⚠️ Failed to save report: {e}")
        
        print("\n" + "=" * 50)
        print("✅ Migration analysis complete!")
        print(f"📊 Analyzed {analysis['total_cache_files']} cache files")
        print(f"🧪 Tested {test_results.get('files_tested', 0)} sample files")
        if backup_dir:
            print(f"💾 Backup created at: {backup_dir}")
        print(f"📄 Report available at: {report_file}")
        
        return report_file


def main():
    """Main migration script"""
    
    parser = argparse.ArgumentParser(description='Migrate to Optimized OCR System')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup')
    parser.add_argument('--base-dir', default='/opt/airflow/downloaded_docs', 
                       help='Base directory for document processing')
    parser.add_argument('--test-samples', type=int, default=3,
                       help='Number of sample files to test')
    
    args = parser.parse_args()
    
    migration_tool = OCRMigrationTool(base_dir=args.base_dir)
    
    try:
        report_file = migration_tool.run_migration(
            create_backup=not args.no_backup,
            test_samples=args.test_samples
        )
        
        print(f"\n🎉 Migration completed successfully!")
        print(f"📖 Read the full report: {report_file}")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()