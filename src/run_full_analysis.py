"""
Full Analysis Pipeline Runner

This is the main entry point for running complete analysis pipeline on a trained model.
Runs all 7 analysis scripts in sequence:
1. evaluate.py - Model evaluation metrics
2. error_analysis.py - Error pattern analysis
3. case_studies.py - Disease-specific case studies
4. visualize_embeddings.py - Embedding visualization
5. explain_predictions.py - Path-based explanations
6. medical_validation.py - Medical validation of predictions
7. compare_methods.py - Method comparison with baselines
8. analyze_failures.py - Failure mode analysis

Usage:
    # Run on final model with final results directory
    python src/run_full_analysis.py --model_path models/final_model.pt --output_dir results_final
    
    # Run on best model (default)
    python src/run_full_analysis.py
    
    # Run specific analyses only
    python src/run_full_analysis.py --analyses evaluate analyze_results case_studies
    
    # Skip certain analyses
    python src/run_full_analysis.py --skip error_analysis embeddings
"""

import os
import sys
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('full_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    Run complete analysis pipeline on a trained model.
    """
    
    # Define all available analyses
    ANALYSES = {
        'evaluate': {
            'script': 'src/evaluate.py',
            'description': 'Model evaluation with metrics',
            'args': '--model_path {model_path} --output_dir {output_dir}',
            'output_subdir': ''
        },
        'error_analysis': {
            'script': 'src/error_analysis.py',
            'description': 'Error pattern analysis',
            'args': '--model_path {model_path} --output_dir {output_dir}/error_analysis',
            'output_subdir': 'error_analysis'
        },
        'case_studies': {
            'script': 'src/case_studies.py',
            'description': 'Disease-specific case studies',
            'args': '--model_path {model_path} --output_dir {output_dir}/case_studies',
            'output_subdir': 'case_studies',
            'diseases': ['diabetes mellitus', 'Alzheimer disease']  # Default diseases
        },
        'embeddings': {
            'script': 'src/visualize_embeddings.py',
            'description': 'Embedding visualization',
            'args': '--model_path {model_path} --output_dir {output_dir}/embeddings --sample_size 5000',
            'output_subdir': 'embeddings'
        },
        'explanations': {
            'script': 'src/explain_predictions.py',
            'description': 'Path-based prediction explanations',
            'args': '--model_path {model_path} --output_dir {output_dir}/explanations',
            'output_subdir': 'explanations',
            'examples': [
                ('Metformin', 'diabetes mellitus'),
                ('Aspirin', 'heart disease')
            ]
        },
        'validation': {
            'script': 'src/medical_validation.py',
            'description': 'Medical validation of predictions',
            'args': '--model_path {model_path} --output_dir {output_dir}/validation --top_k 50 --sample_diseases 100',
            'output_subdir': 'validation'
        },
        'comparison': {
            'script': 'src/compare_methods.py',
            'description': 'Method comparison with baselines',
            'args': '--model_path {model_path} --output_dir {output_dir}/comparison --methods random degree rgcn',
            'output_subdir': 'comparison'
        },
        'failures': {
            'script': 'src/analyze_failures.py',
            'description': 'Failure mode analysis',
            'args': '--model_path {model_path} --output_dir {output_dir}/failure_analysis --num_failures 5 --num_successes 5 --visualize_subgraphs',
            'output_subdir': 'failure_analysis'
        }
    }
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = 'results',
        data_dir: str = 'data/processed',
        python_path: str = './venv/bin/python'
    ):
        """
        Initialize analysis pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            output_dir: Output directory for all results
            data_dir: Directory containing processed data
            python_path: Path to Python interpreter
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.python_path = python_path
        
        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized pipeline:")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Data: {self.data_dir}")
    
    def run_analysis(
        self,
        analysis_name: str,
        timeout: Optional[int] = None,
        additional_args: Optional[str] = None
    ) -> bool:
        """
        Run a single analysis script.
        
        Args:
            analysis_name: Name of analysis to run
            timeout: Maximum execution time in seconds (None = no limit)
            additional_args: Additional command-line arguments
            
        Returns:
            True if successful, False otherwise
        """
        if analysis_name not in self.ANALYSES:
            logger.error(f"Unknown analysis: {analysis_name}")
            return False
        
        analysis = self.ANALYSES[analysis_name]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Running: {analysis_name.upper()}")
        logger.info(f"Description: {analysis['description']}")
        logger.info(f"{'='*80}\n")
        
        # Create output subdirectory
        if analysis['output_subdir']:
            subdir = self.output_dir / analysis['output_subdir']
            subdir.mkdir(parents=True, exist_ok=True)
        
        # Format arguments
        args_str = analysis['args'].format(
            model_path=self.model_path,
            output_dir=self.output_dir
        )
        
        # Handle special cases
        if analysis_name == 'case_studies' and 'diseases' in analysis:
            # Run case studies for each disease
            for disease in analysis['diseases']:
                disease_args = f"{args_str} --disease \"{disease}\""
                if additional_args:
                    disease_args += f" {additional_args}"
                
                cmd = f"{self.python_path} {analysis['script']} {disease_args}"
                logger.info(f"Command: {cmd}")
                
                if not self._run_command(cmd, timeout):
                    logger.warning(f"Failed case study for: {disease}")
        
        elif analysis_name == 'explanations' and 'examples' in analysis:
            # Run explanations for each example
            for drug, disease in analysis['examples']:
                example_args = f"{args_str} --drug \"{drug}\" --disease \"{disease}\" --top_k 5"
                if additional_args:
                    example_args += f" {additional_args}"
                
                cmd = f"{self.python_path} {analysis['script']} {example_args}"
                logger.info(f"Command: {cmd}")
                
                if not self._run_command(cmd, timeout):
                    logger.warning(f"Failed explanation for: {drug} → {disease}")
        
        else:
            # Standard analysis
            if additional_args:
                args_str += f" {additional_args}"
            
            cmd = f"{self.python_path} {analysis['script']} {args_str}"
            logger.info(f"Command: {cmd}")
            
            if not self._run_command(cmd, timeout):
                logger.error(f"Failed to run {analysis_name}")
                return False
        
        logger.info(f"✓ Completed: {analysis_name}")
        return True
    
    def _run_command(self, cmd: str, timeout: Optional[int] = None) -> bool:
        """
        Execute a shell command.
        
        Args:
            cmd: Command to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Completed in {elapsed:.1f}s")
            
            # Log output if there are important messages
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if any(keyword in line.lower() for keyword in ['error', 'warning', 'saved', 'completed']):
                        logger.info(line)
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s")
            return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"Output: {e.stdout[-500:]}")  # Last 500 chars
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return False
    
    def run_all(
        self,
        analyses: Optional[List[str]] = None,
        skip: Optional[List[str]] = None,
        timeout: Optional[int] = 300
    ) -> dict:
        """
        Run all or selected analyses.
        
        Args:
            analyses: List of specific analyses to run (None = all)
            skip: List of analyses to skip
            timeout: Maximum time per analysis in seconds
            
        Returns:
            Dictionary with results for each analysis
        """
        # Determine which analyses to run
        if analyses:
            to_run = [a for a in analyses if a in self.ANALYSES]
        else:
            to_run = list(self.ANALYSES.keys())
        
        if skip:
            to_run = [a for a in to_run if a not in skip]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING FULL ANALYSIS PIPELINE")
        logger.info(f"{'='*80}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Analyses to run: {', '.join(to_run)}")
        logger.info(f"{'='*80}\n")
        
        results = {}
        start_time = time.time()
        
        for analysis_name in to_run:
            analysis_start = time.time()
            
            try:
                success = self.run_analysis(analysis_name, timeout=timeout)
                results[analysis_name] = {
                    'success': success,
                    'duration': time.time() - analysis_start
                }
            except Exception as e:
                logger.error(f"Error running {analysis_name}: {str(e)}")
                results[analysis_name] = {
                    'success': False,
                    'duration': time.time() - analysis_start,
                    'error': str(e)
                }
        
        total_time = time.time() - start_time
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"PIPELINE SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"\nResults:")
        
        successful = 0
        failed = 0
        
        for name, result in results.items():
            status = "✓" if result['success'] else "✗"
            duration = result['duration']
            logger.info(f"  {status} {name:20s} ({duration:.1f}s)")
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        logger.info(f"\nSuccessful: {successful}/{len(results)}")
        if failed > 0:
            logger.warning(f"Failed: {failed}/{len(results)}")
        
        logger.info(f"\nAll results saved to: {self.output_dir}")
        logger.info(f"{'='*80}\n")
        
        return results
    
    def list_analyses(self):
        """List all available analyses."""
        print("\nAvailable analyses:\n")
        for name, info in self.ANALYSES.items():
            print(f"  {name:20s} - {info['description']}")
        print()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run complete analysis pipeline on trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all analyses on final model
    python src/run_full_analysis.py --model_path models/final_model.pt --output_dir results_final
    
    # Run all analyses on best model (default)
    python src/run_full_analysis.py
    
    # Run specific analyses only
    python src/run_full_analysis.py --analyses evaluate analyze_results case_studies
    
    # Skip certain analyses
    python src/run_full_analysis.py --skip embeddings explanations
    
    # List all available analyses
    python src/run_full_analysis.py --list
    
    # Custom timeout
    python src/run_full_analysis.py --timeout 600
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='output/models/best_model.pt',
        help='Path to trained model checkpoint (default: output/models/best_model.pt)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for all results (default: results)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data (default: data/processed)'
    )
    
    parser.add_argument(
        '--python_path',
        type=str,
        default='./venv/bin/python',
        help='Path to Python interpreter (default: ./venv/bin/python)'
    )
    
    parser.add_argument(
        '--analyses',
        nargs='+',
        choices=list(AnalysisPipeline.ANALYSES.keys()),
        help='Specific analyses to run (default: all)'
    )
    
    parser.add_argument(
        '--skip',
        nargs='+',
        choices=list(AnalysisPipeline.ANALYSES.keys()),
        help='Analyses to skip'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Maximum time per analysis in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available analyses and exit'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # List analyses if requested
    if args.list:
        pipeline = AnalysisPipeline(
            model_path=args.model_path if os.path.exists(args.model_path) else 'output/models/best_model.pt',
            output_dir=args.output_dir
        )
        pipeline.list_analyses()
        sys.exit(0)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        logger.info(f"Please specify correct model path with --model_path")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = AnalysisPipeline(
            model_path=args.model_path,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            python_path=args.python_path
        )
        
        # Run analyses
        results = pipeline.run_all(
            analyses=args.analyses,
            skip=args.skip,
            timeout=args.timeout
        )
        
        # Check if all succeeded
        all_success = all(r['success'] for r in results.values())
        
        if all_success:
            logger.info("✓ All analyses completed successfully!")
            sys.exit(0)
        else:
            logger.warning("⚠ Some analyses failed. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
