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


def get_l2_directories(noise_level):
    """
    Find all L2 regularization directories for a given noise level
    
    Args:
        noise_level (float): Noise level to match
    
    Returns:
        str: Path to the L2 directory
    """
    l2_dir_name = f"../../Results_{noise_level}_L2"
    
    if os.path.isdir(l2_dir_name):
        return l2_dir_name
    
    return None


def find_l2_experiments(l2_dir, dataset):
    """
    Find all L2 experiments for a dataset, grouped by L2 values
    
    Args:
        l2_dir (str): Directory to search in
        dataset (str): Dataset name to filter by
    
    Returns:
        dict: Dictionary mapping L2 values to experiment paths
    """
    l2_experiments = {}
    
    if not os.path.isdir(l2_dir):
        return l2_experiments
        
    for experiment in os.listdir(l2_dir):
        experiment_path = os.path.join(l2_dir, experiment)
        
        # Skip if not a directory or is a comparison directory
        if not os.path.isdir(experiment_path) or experiment.startswith('!'):
            continue
        
        # Check if it's an L2 experiment and contains dataset name
        if not (dataset.lower() in experiment.lower() and 'reg_L2' in experiment):
            continue
        
        # Extract L2 value
        try:
            # Assuming format like "base_cifar10_reg_L2_0.01"
            l2_value = float(experiment.split('_')[-1])
            
            # Add to dictionary
            if l2_value not in l2_experiments:
                l2_experiments[l2_value] = []
            
            l2_experiments[l2_value].append({
                'name': experiment,
                'path': experiment_path
            })
        except (ValueError, IndexError):
            # Skip if L2 value cannot be extracted
            continue
    
    return l2_experiments


def analyze_l2_experiments(noise_level, dataset):
    """
    Analyze L2 experiments and identify the best L2 value
    
    Args:
        noise_level (float): Noise level to analyze
        dataset (str): Dataset name to filter by
        
    Returns:
        tuple: (best_l2_value, best_experiment_data)
    """
    # Find L2 directory
    l2_dir = get_l2_directories(noise_level)
    
    if not l2_dir:
        print(f"No L2 directory found for noise level {noise_level}")
        return None, None
    
    # Find L2 experiments
    l2_experiments = find_l2_experiments(l2_dir, dataset)
    
    if not l2_experiments:
        print(f"No L2 experiments found for {dataset} in {l2_dir}")
        return None, None
    
    # Analyze experiments for each L2 value
    l2_results = {}
    
    for l2_value, experiments in l2_experiments.items():
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
            l2_results[l2_value] = best_experiment
    
    if not l2_results:
        print(f"No valid results found for {dataset} L2 experiments in {l2_dir}")
        return None, None
    
    # Find overall best L2 value (highest accuracy, then lowest loss)
    best_l2_value = None
    best_accuracy = 0
    best_loss = float('inf')
    
    for l2_value, exp_data in l2_results.items():
        test_accuracy = exp_data['test_accuracy']
        test_loss = exp_data['test_loss']
        
        is_better = (test_accuracy > best_accuracy) or \
                   (test_accuracy == best_accuracy and test_loss < best_loss)
        
        if is_better:
            best_accuracy = test_accuracy
            best_loss = test_loss
            best_l2_value = l2_value
    
    return best_l2_value, l2_results[best_l2_value] if best_l2_value is not None else None


def save_best_experiment(noise_level, dataset, best_l2_value, best_experiment_data):
    """
    Save best L2 experiment to a new directory
    
    Args:
        noise_level (float): Noise level
        dataset (str): Dataset name
        best_l2_value (float): Best L2 value
        best_experiment_data (dict): Data for the best experiment
    """
    if best_l2_value is None or best_experiment_data is None:
        return
    
    # Create destination directory
    l2_dir = get_l2_directories(noise_level)
    best_dir_name = f"!Best_L2_{dataset.upper()}_{noise_level}"
    best_dir_path = os.path.join(l2_dir, best_dir_name)
    
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
        'best_l2_value': best_l2_value,
        'experiment_name': best_experiment_data['name'],
        'test_accuracy': best_experiment_data['test_accuracy'],
        'test_loss': best_experiment_data['test_loss'],
        'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(best_dir_path, 'best_l2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Saved best L2 experiment for {dataset} (noise {noise_level}) to {best_dir_path}")
    print(f"  Best L2 value: {best_l2_value}")
    print(f"  Test accuracy: {best_experiment_data['test_accuracy']:.4f}")
    print(f"  Test loss: {best_experiment_data['test_loss']:.4f}")


def create_comparison_report(noise_level, dataset, l2_experiments):
    """
    Create a comparison report for all L2 values
    
    Args:
        noise_level (float): Noise level
        dataset (str): Dataset name
        l2_experiments (dict): Mapping of L2 values to experiment data
    """
    if not l2_experiments:
        return
    
    # Find L2 directory
    l2_dir = get_l2_directories(noise_level)
    
    # Create report directory
    report_dir = os.path.join(l2_dir, f"!L2_Compare_{dataset.upper()}_{noise_level}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Prepare comparison data
    comparison_data = []
    
    for l2_value, exp_data in l2_experiments.items():
        comparison_data.append({
            'L2_Value': l2_value,
            'Experiment_Name': exp_data['name'],
            'Test_Accuracy': exp_data['test_accuracy'],
            'Test_Loss': exp_data['test_loss']
        })
    
    # Create comparison CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_Accuracy', ascending=False)
    comparison_df.to_csv(os.path.join(report_dir, 'l2_comparison.csv'), index=False)
    
    # Create summary text file
    with open(os.path.join(report_dir, 'l2_summary.txt'), 'w') as f:
        f.write(f"L2 Regularization Comparison for {dataset.upper()} (Noise Level: {noise_level})\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("L2 Values sorted by Test Accuracy (highest to lowest):\n")
        f.write("-" * 80 + "\n")
        
        for _, row in comparison_df.iterrows():
            f.write(f"L2 Value: {row['L2_Value']:.6f}\n")
            f.write(f"  Experiment: {row['Experiment_Name']}\n")
            f.write(f"  Test Accuracy: {row['Test_Accuracy']:.6f}\n")
            f.write(f"  Test Loss: {row['Test_Loss']:.6f}\n")
            f.write("-" * 80 + "\n")
        
        f.write("\nReport generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    
    print(f"Created comparison report at {report_dir}")


def main():
    """
    Main function to analyze L2 regularization experiments
    """
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Analyze L2 regularization experiment results')
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
        print(f"Processing L2 experiments for noise level: {noise}")
        print(f"{'='*75}")
        
        for dataset in datasets_to_process:
            print(f"\n{'-'*75}")
            print(f"DATASET: {dataset.upper()}, NOISE LEVEL: {noise}")
            print(f"{'-'*75}")
            
            # Get L2 experiments and find the best one
            best_l2_value, best_experiment = analyze_l2_experiments(noise, dataset)
            
            if best_l2_value is not None:
                # Save best experiment to a new directory
                save_best_experiment(noise, dataset, best_l2_value, best_experiment)
                
                # Get all L2 experiment data for comparison report
                l2_dir = get_l2_directories(noise)
                l2_experiments = find_l2_experiments(l2_dir, dataset)
                
                # Create a dictionary of best experiments for each L2 value
                best_by_l2 = {}
                for l2_value, exps in l2_experiments.items():
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
                        best_by_l2[l2_value] = best_exp
                
                # Create comparison report
                create_comparison_report(noise, dataset, best_by_l2)
            else:
                print(f"No valid L2 experiments found for {dataset} (noise {noise})")
    
    print(f"\n{'='*75}")
    print("L2 regularization analysis complete!")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()