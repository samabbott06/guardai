"""
Main script for running the indirect prompt injection simulation framework.
This coordinates loading prompts, running apps, and generating reports.
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Optional
import time

# Import application classes
from synthetic_llm_app import SyntheticLLMApp
from email_manager_app import EmailManagerApp
from tech_support_app import TechSupportChatbotApp
from document_summarizer_app import DocumentSummarizerApp
from personal_assistant_app import PersonalAssistantApp
from evaluator import Evaluator

def setup_args():
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(description='Run indirect prompt injection simulations')
    
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing prompt CSV files')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save log files')
    parser.add_argument('--report_path', type=str, default='report.json',
                        help='Path to save the evaluation report')
    parser.add_argument('--apps', type=str, nargs='+', 
                        choices=['email', 'techsupport', 'document', 'assistant', 'all'],
                        default=['all'],
                        help='Which apps to run simulations for')
    parser.add_argument('--attacks', type=str, nargs='+',
                        choices=['information_gathering', 'intrusion', 'malware', 'manipulated_content', 'all'],
                        default=['all'],
                        help='Which attack types to test')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of prompts per category (for quick testing)')
                        
    return parser.parse_args()

def load_model():
    """
    Load the LLM model to be shared across apps.
    Returns a LangChain LLM object.
    """
    # Return None for now - each app will initialize its own model if needed
    return None

def get_app_mapping():
    """
    Get mapping of attack types to app classes.
    Returns a dictionary mapping attack type to appropriate app class.
    """
    return {
        "information_gathering": [EmailManagerApp, PersonalAssistantApp],    # Email app and Personal Assistant for data theft scenarios
        "intrusion": [TechSupportChatbotApp, PersonalAssistantApp],          # Tech support and Personal Assistant for system intrusion
        "malware": [EmailManagerApp],                                        # Email app for malware scenarios
        "manipulated_content": [DocumentSummarizerApp]                       # Document summarizer for content manipulation
    }

def discover_csv_files(data_dir: str, attacks: List[str]) -> Dict[str, str]:
    """
    Discover CSV files in the data directory.
    
    Args:
        data_dir: Directory containing CSV files
        attacks: List of attack types to include
        
    Returns:
        Dictionary mapping attack type to CSV file path
    """
    csv_files = {}
    
    # Handle 'all' option
    if 'all' in attacks:
        attacks = ['information_gathering', 'intrusion', 'malware', 'manipulated_content']
    
    # Look for CSV files matching attack types
    for attack in attacks:
        csv_path = os.path.join(data_dir, f"{attack}.csv")
        if os.path.exists(csv_path):
            csv_files[attack] = csv_path
        else:
            print(f"Warning: Could not find CSV file for {attack} attack type")
    
    return csv_files

def run_simulations(
    csv_files: Dict[str, str],
    app_mapping: Dict[str, type],
    log_dir: str,
    apps: List[str],
    prompt_limit: Optional[int] = None
) -> List[Dict]:
    """
    Run simulations for each attack type using the appropriate app.
    
    Args:
        csv_files: Dictionary mapping attack type to CSV file path
        app_mapping: Dictionary mapping attack type to app class
        log_dir: Directory to save log files
        apps: List of app types to run
        prompt_limit: Maximum number of prompts to process per category
        
    Returns:
        List of all log entries from simulations
    """
    all_logs = []
    shared_model = load_model()
    
    # Handle 'all' option for apps
    if 'all' in apps:
        # Collect all unique app classes from app_mapping
        selected_apps = set()
        for app_classes in app_mapping.values():
            for app_class in app_classes:
                selected_apps.add(app_class)
    else:
        # Convert app names to app classes
        app_name_to_class = {
            'email': EmailManagerApp,
            'techsupport': TechSupportChatbotApp,
            'document': DocumentSummarizerApp,
            'assistant': PersonalAssistantApp
        }
        selected_apps = set(app_name_to_class[app] for app in apps if app in app_name_to_class)
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Run simulations for each attack type and app combination
    for attack_type, csv_path in csv_files.items():
        print(f"\n=== Running simulations for {attack_type} attack type ===")
        
        # Get default apps for this attack type
        app_classes = app_mapping.get(attack_type, [])
        if not app_classes:
            print(f"No apps mapped for {attack_type}, skipping")
            continue
        
        # Determine which apps to run for this attack type
        apps_to_run = []
        for app_class in app_classes:
            if app_class in selected_apps:
                apps_to_run.append(app_class)
        
        if not apps_to_run:
            print(f"No selected apps available for {attack_type}, skipping")
            continue
        
        for app_class in apps_to_run:
            app_name = app_class.__name__
            log_path = os.path.join(log_dir, f"{attack_type}_{app_name}.jsonl")
            
            print(f"Running {app_name} for {attack_type}")
            
            # Initialize app instance
            app = app_class(model=shared_model, log_path=log_path)
            
            # Load prompts
            prompts = app.load_prompts(csv_path)
            
            # Apply limit if specified
            if prompt_limit is not None and prompt_limit > 0:
                prompts = prompts[:prompt_limit]
                print(f"Limited to {len(prompts)} prompts")
            
            # Process each prompt
            for i, prompt_data in enumerate(prompts):
                print(f"Processing prompt {i+1}/{len(prompts)}")
                
                try:
                    # Run the prompt through the app
                    output, compromised, flags = app.run_prompt(prompt_data)
                    
                    # Add a brief pause to avoid overwhelming the model
                    time.sleep(0.5)
                    
                    # Print status
                    status = "COMPROMISED" if compromised else "OK"
                    print(f"  Result: {status} - Flags: {', '.join(flags) if flags else 'None'}")
                    
                except Exception as e:
                    print(f"  Error processing prompt: {e}")
            
            # Save logs
            app.save_logs()
            
            # Collect logs for report
            all_logs.extend(app.log_records)
            
            print(f"Completed {app_name} for {attack_type}")
    
    return all_logs

def generate_report(logs: List[Dict], report_path: str):
    """
    Generate and save evaluation report.
    
    Args:
        logs: List of log entries from simulations
        report_path: Path to save the report
    """
    # Use the Evaluator to generate a report
    evaluator = Evaluator()
    report = evaluator.generate_report(logs)
    
    # Save the report
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # Print a summary
    print("\n=== Evaluation Summary ===")
    print(f"Total prompts: {report['total_prompts']}")
    print(f"Compromised: {report['total_compromised']} ({report['compromise_rate']:.1f}%)")
    
    print("\nResults by attack type:")
    for attack_type, stats in report["by_attack_type"].items():
        print(f"  {attack_type}: {stats['compromised']}/{stats['total']} compromised ({stats['compromise_rate']:.1f}%)")
    
    print("\nResults by app:")
    for app_name, stats in report["by_app"].items():
        print(f"  {app_name}: {stats['compromised']}/{stats['total']} compromised ({stats['compromise_rate']:.1f}%)")
    
    print(f"\nFull report saved to {report_path}")

def main():
    """Main function to run the simulation framework."""
    args = setup_args()
    
    print("=== Indirect Prompt Injection Simulation Framework ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Log directory: {args.log_dir}")
    
    # Discover CSV files
    csv_files = discover_csv_files(args.data_dir, args.attacks)
    if not csv_files:
        print("Error: No CSV files found for the specified attack types")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Get app mapping
    app_mapping = get_app_mapping()
    
    # Run simulations
    logs = run_simulations(
        csv_files=csv_files,
        app_mapping=app_mapping,
        log_dir=args.log_dir,
        apps=args.apps,
        prompt_limit=args.limit
    )
    
    # Generate report
    generate_report(logs, args.report_path)
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()