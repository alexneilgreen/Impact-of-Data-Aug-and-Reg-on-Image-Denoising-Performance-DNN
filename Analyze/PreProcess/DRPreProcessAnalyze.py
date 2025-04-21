import os
import json
import pandas as pd
import shutil
import argparse
from datetime import datetime


def load_experiment_results(experiment_path):
    """
    Load results from JSON and CSV files in an experiment folder
    
    Args:
        experiment_path (str): Path to experiment folder
    
    Returns:
        dict: Loaded results
    """
    results = {}
    
    # Try to load JSON results
    json_path = os.path.join(experiment_path, 'results.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
    
    # Check for CSV files
    csv_path = os.path.join(experiment_path, 'epoch_data.csv')
    if os.path.exists(csv_path):
        results['epoch_data'] = pd.read_csv(csv_path)
    
    return results


def get_dr_directories(noise_level):
    """
    Find all Dropout regularization directories for a given noise level
    
    Args:
        noise_level (float): Noise level to match
    
    Returns:
        str: Path to the DR directory
    """
    dr_dir_name = f"../../Results_{noise_level}_DR"
    
    if os.path.isdir(dr_dir_name):
        return dr_dir_name
    
    return None


def find_dr_experiments(dr_dir, dataset):
    """
    Find all Dropout experiments for a dataset, grouped by dropout rates
    
    Args:
        dr_dir (str): Directory to search in
        dataset (str): Dataset name to filter by
    
    Returns:
        dict: Dictionary mapping dropout rates to experiment paths
    """
    dr_experiments = {}
    
    if not os.path.isdir(dr_dir):
        return dr_experiments
        
    for experiment in os.listdir(dr_dir):
        experiment_path = os.path.join(dr_dir, experiment)
        
        # Skip if not a directory or is a comparison directory
        if not os.path.isdir(experiment_path) or experiment.startswith('!'):
            continue
        
        # Check if it's a DR experiment and contains dataset name
        if not (dataset.lower() in experiment.lower() and 'reg_DR' in experiment):
            continue
        
        # Extract dropout rate
        try:
            # Assuming format like "base_cifar10_reg_DR_0.2"
            dr_value = float(experiment.split('_')[-1])
            
            # Add to dictionary
            if dr_value not in dr_experiments:
                dr_experiments[dr_value] = []
            
            dr_experiments[dr_value].append({
                'name': experiment,
                'path': experiment_path
            })
        except (ValueError, IndexError):
            # Skip if dropout rate cannot be extracted
            continue
    
    return dr_experiments


def analyze_dr_experiments(noise_level, dataset):
    """
    Analyze Dropout experiments and identify the best dropout rate
    
    Args:
        noise_level (float): Noise level to analyze
        dataset (str): Dataset name to filter by
        
    Returns:
        tuple: (best_dr_value, best_experiment_data)
    """
    # Find DR directory
    dr_dir = get_dr_directories(noise_level)
    
    if not dr_dir:
        print(f"No Dropout directory found for noise level {noise_level}")
        return None, None
    
    # Find DR experiments
    dr_experiments = find_dr_experiments(dr_dir, dataset)
    
    if not dr_experiments:
        print(f"No Dropout experiments found for {dataset} in {dr_dir}")
        return None, None
    
    # Analyze experiments for each dropout rate
    dr_results = {}
    
    for dr_value, experiments in dr_experiments.items():
        best_accuracy = 0
        best_loss = float('inf')
        best_experiment = None
        
        for exp in experiments:
            results = load_experiment_results(exp['path'])
            
            if not results or 'test_accuracy' not in results or 'test_loss' not in results:
                continue
            
            test_accuracy = results['test_accuracy']
            test_loss = results['test_loss']
            
            # Prioritize accuracy over loss if they're different
            is_better = (test_accuracy > best_accuracy) or \
                       (test_accuracy == best_accuracy and test_loss < best_loss)
            
            if is_better:
                best_accuracy = test_accuracy
                best_loss = test_loss
                best_experiment = {
                    'name': exp['name'],
                    'path': exp['path'],
                    'test_accuracy': test_accuracy,
                    'test_loss': test_loss,
                    'results': results
                }
        
        if best_experiment:
            dr_results[dr_value] = best_experiment
    
    if not dr_results:
        print(f"No valid results found for {dataset} Dropout experiments in {dr_dir}")
        return None, None
    
    # Find overall best dropout rate (highest accuracy, then lowest loss)
    best_dr_value = None
    best_accuracy = 0
    best_loss = float('inf')
    
    for dr_value, exp_data in dr_results.items():
        test_accuracy = exp_data['test_accuracy']
        test_loss = exp_data['test_loss']
        
        is_better = (test_accuracy > best_accuracy) or \
                   (test_accuracy == best_accuracy and test_loss < best_loss)
        
        if is_better:
            best_accuracy = test_accuracy
            best_loss = test_loss
            best_dr_value = dr_value
    
    return best_dr_value, dr_results[best_dr_value] if best_dr_value is not None else None


def save_best_experiment(noise_level, dataset, best_dr_value, best_experiment_data):
    """
    Save best Dropout experiment to a new directory
    
    Args:
        noise_level (float): Noise level
        dataset (str): Dataset name
        best_dr_value (float): Best dropout rate
        best_experiment_data (dict): Data for the best experiment
    """
    if best_dr_value is None or best_experiment_data is None:
        return
    
    # Create destination directory
    dr_dir = get_dr_directories(noise_level)
    best_dir_name = f"!Best_DR_{dataset.upper()}_{noise_level}"
    best_dir_path = os.path.join(dr_dir, best_dir_name)
    
    # Create directory if it doesn't exist
    if os.path.exists(best_dir_path):
        # Delete existing directory
        shutil.rmtree(best_dir_path)
    
    os.makedirs(best_dir_path, exist_ok=True)
    
    # Copy experiment files
    src_path = best_experiment_data['path']
    for item in os.listdir(src_path):
        s = os.path.join(src_path, item)
        d = os.path.join(best_dir_path, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    
    # Create summary file
    summary = {
        'dataset': dataset,
        'noise_level': noise_level,
        'best_dropout_rate': best_dr_value,
        'experiment_name': best_experiment_data['name'],
        'test_accuracy': best_experiment_data['test_accuracy'],
        'test_loss': best_experiment_data['test_loss'],
        'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(best_dir_path, 'best_dr_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Saved best Dropout experiment for {dataset} (noise {noise_level}) to {best_dir_path}")
    print(f"  Best dropout rate: {best_dr_value}")
    print(f"  Test accuracy: {best_experiment_data['test_accuracy']:.4f}")
    print(f"  Test loss: {best_experiment_data['test_loss']:.4f}")


def create_comparison_report(noise_level, dataset, dr_experiments):
    """
    Create a comparison report for all dropout rates
    
    Args:
        noise_level (float): Noise level
        dataset (str): Dataset name
        dr_experiments (dict): Mapping of dropout rates to experiment data
    """
    if not dr_experiments:
        return
    
    # Find DR directory
    dr_dir = get_dr_directories(noise_level)
    
    # Create report directory
    report_dir = os.path.join(dr_dir, f"!DR_Compare_{dataset.upper()}_{noise_level}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Prepare comparison data
    comparison_data = []
    
    for dr_value, exp_data in dr_experiments.items():
        comparison_data.append({
            'Dropout_Rate': dr_value,
            'Experiment_Name': exp_data['name'],
            'Test_Accuracy': exp_data['test_accuracy'],
            'Test_Loss': exp_data['test_loss']
        })
    
    # Create comparison CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_Accuracy', ascending=False)
    comparison_df.to_csv(os.path.join(report_dir, 'dr_comparison.csv'), index=False)
    
    # Create summary text file
    with open(os.path.join(report_dir, 'dr_summary.txt'), 'w') as f:
        f.write(f"Dropout Regularization Comparison for {dataset.upper()} (Noise Level: {noise_level})\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Dropout Rates sorted by Test Accuracy (highest to lowest):\n")
        f.write("-" * 80 + "\n")
        
        for _, row in comparison_df.iterrows():
            f.write(f"Dropout Rate: {row['Dropout_Rate']:.6f}\n")
            f.write(f"  Experiment: {row['Experiment_Name']}\n")
            f.write(f"  Test Accuracy: {row['Test_Accuracy']:.6f}\n")
            f.write(f"  Test Loss: {row['Test_Loss']:.6f}\n")
            f.write("-" * 80 + "\n")
        
        f.write("\nReport generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    
    print(f"Created comparison report at {report_dir}")


def main():
    """
    Main function to analyze Dropout regularization experiments
    """
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Analyze Dropout regularization experiment results')
    parser.add_argument('--noise', type=str, default='all',
                      help='Noise level for results directory (choices: 0.15, 0.25, 0.5, all) (default: all)')
    parser.add_argument('--dataset', type=str, default='all',
                     help='Dataset to use (choices: MNIST, CIFAR10, CIFAR100, STL10, all) (default: all)')
    args = parser.parse_args()
    
    # Define available noise levels and datasets
    noise_levels = [0.15, 0.25, 0.5]
    datasets = ['MNIST', 'CIFAR10', 'CIFAR100', 'STL10']
    
    # Determine which noise levels to process
    if args.noise.lower() == 'all':
        noise_to_process = noise_levels
    else:
        try:
            noise_value = float(args.noise)
            if noise_value in noise_levels:
                noise_to_process = [noise_value]
            else:
                print(f"Invalid noise value: {args.noise}. Using default 'all'.")
                noise_to_process = noise_levels
        except ValueError:
            print(f"Invalid noise value: {args.noise}. Using default 'all'.")
            noise_to_process = noise_levels
    
    # Determine which datasets to process
    if args.dataset.lower() == 'all':
        datasets_to_process = datasets
    elif args.dataset in datasets:
        datasets_to_process = [args.dataset]
    else:
        print(f"Invalid dataset: {args.dataset}. Using default 'all'.")
        datasets_to_process = datasets
    
    # Process each noise level and dataset
    for noise in noise_to_process:
        print(f"\n{'='*75}")
        print(f"Processing Dropout experiments for noise level: {noise}")
        print(f"{'='*75}")
        
        for dataset in datasets_to_process:
            print(f"\n{'-'*75}")
            print(f"DATASET: {dataset.upper()}, NOISE LEVEL: {noise}")
            print(f"{'-'*75}")
            
            # Get DR experiments and find the best one
            best_dr_value, best_experiment = analyze_dr_experiments(noise, dataset)
            
            if best_dr_value is not None:
                # Save best experiment to a new directory
                save_best_experiment(noise, dataset, best_dr_value, best_experiment)
                
                # Get all DR experiment data for comparison report
                dr_dir = get_dr_directories(noise)
                dr_experiments = find_dr_experiments(dr_dir, dataset)
                
                # Create a dictionary of best experiments for each dropout rate
                best_by_dr = {}
                for dr_value, exps in dr_experiments.items():
                    best_acc = 0
                    best_loss = float('inf')
                    best_exp = None
                    
                    for exp in exps:
                        results = load_experiment_results(exp['path'])
                        
                        if not results or 'test_accuracy' not in results or 'test_loss' not in results:
                            continue
                        
                        if results['test_accuracy'] > best_acc or (results['test_accuracy'] == best_acc and results['test_loss'] < best_loss):
                            best_acc = results['test_accuracy']
                            best_loss = results['test_loss']
                            best_exp = {
                                'name': exp['name'],
                                'path': exp['path'],
                                'test_accuracy': results['test_accuracy'],
                                'test_loss': results['test_loss'],
                                'results': results
                            }
                    
                    if best_exp:
                        best_by_dr[dr_value] = best_exp
                
                # Create comparison report
                create_comparison_report(noise, dataset, best_by_dr)
            else:
                print(f"No valid Dropout experiments found for {dataset} (noise {noise})")
    
    print(f"\n{'='*75}")
    print("Dropout regularization analysis complete!")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()