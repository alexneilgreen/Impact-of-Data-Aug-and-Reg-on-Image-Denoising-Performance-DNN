import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

from matplotlib.patches import Patch

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

def create_epoch_plots(results, experiment_path):
    """
    Create plots for training, validation, and test metrics
    
    Args:
        results (dict): Experiment results
        experiment_path (str): Path to experiment folder
    """    
    # Check if epoch data exists
    if 'epoch_data' not in results:
        print(f"No epoch data found for {experiment_path}")
        return
    
    df = results['epoch_data']
    
    plt.figure(figsize=(15, 6))
    
    # Loss Subplot
    plt.subplot(1, 3, 1)
    plt.title('Loss per Epoch')
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Subplot
    plt.subplot(1, 3, 2)
    plt.title('Accuracy per Epoch')
    plt.plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy')
    plt.plot(df['Epoch'], df['Validation Accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Metrics Comparison Subplot
    plt.subplot(1, 3, 3)
    metrics = {
        'Test Loss': results.get('test_loss', 0),
        'Test Accuracy': results.get('test_accuracy', 0)
    }
    
    # Create bar plot
    bars = plt.bar(list(metrics.keys()), list(metrics.values()))
    plt.title('Final Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Value')
    
    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, 
                    f'{height:.4f}', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Add computation time as text at the bottom
    computation_time = results.get('computation_time', 0)
    plt.figtext(0.5, 0.01, 
        f'Computation Time: {int(computation_time // 3600)} Hrs {int((computation_time % 3600) // 60)} Min {int(computation_time % 60)} Sec', 
        ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    # Save the plot
    plot_path = os.path.join(experiment_path, 'epoch_analysis.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    
    plt.close()

def compare_all_experiments(comparison_data, results_dir, dataset):
    """
    Create a comprehensive comparison of all experiments
    Saves results in a dataset-specific directory
    
    Args:
        comparison_data (list): List of experiment metrics
        results_dir (str): Directory containing experiment results
        dataset (str): Dataset to filter experiments by
    """
    # Ensure dataset-specific !Compare_All directory exists
    compare_all_path = os.path.join(results_dir, f'!Compare_All_{dataset.lower()}')
    os.makedirs(compare_all_path, exist_ok=True)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Clean experiment names by removing dataset suffix and formatting
    dataset_suffix = f'_{dataset.lower()}'
    comparison_df['CleanName'] = comparison_df['Experiment'].apply(
        lambda x: ' '.join(word.capitalize() for word in x.replace(dataset_suffix, '').split('_'))
    )
    
    # Plotting individual metrics
    metrics = ['Test Loss', 'Test Accuracy', 'Computation Time']
    
    # For certain metrics, lower is better
    lower_is_better = {'Test Loss': True, 'Computation Time': True}
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Determine top 3 experiments based on metric
        if metric in lower_is_better and lower_is_better[metric]:
            top_indices = comparison_df[metric].nsmallest(3).index
        else:
            top_indices = comparison_df[metric].nlargest(3).index
        
        # Create a list of colors for the bars
        colors = ['#1f77b4'] * len(comparison_df)  # Default blue color
        for idx in top_indices:
            colors[idx] = '#2ca02c'  # Green color for top performers
        
        # Create bar plot with cleaned names and custom colors
        bars = plt.bar(comparison_df['CleanName'], comparison_df[metric], color=colors)
        
        plt.title(f'{metric} Comparison for {dataset}')
        fig_path = os.path.join(compare_all_path, f'{metric.lower().replace(" ", "_")}_comparison_{dataset.lower()}.png')
        
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Experiment Type')
        plt.ylabel(metric)
        
        # Add value annotations
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, 
                    f'{height:.4f}', 
                    ha='center', va='bottom')
        
        # Add a legend indicating green bars are best 3 performers
        legend_elements = [Patch(facecolor='#2ca02c', label='Best 3')]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    
    # Save comparison data with clean names
    csv_path = os.path.join(compare_all_path, f'experiments_comparison_{dataset.lower()}.csv')
    comparison_df.to_csv(csv_path, index=False)

def main():
    """
    Analyze results for all experiments
    """
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--noise', type=str, default='all',
                      help='Noise level for results directory (choices: 0.15, 0.25, 0.5, all) (default: all)')
    parser.add_argument('--dataset', type=str, default='all',
                     help='Dataset to use (choices: MNIST, CIFAR10, CIFAR100, STL10, all) (default: all)')
    parser.add_argument('--regVal', type=str, default=None,
                        choices=['L1', 'L2', 'Dr', 'ES', 'None'],
                        help='Regularization type: L1 (L1 Regularization), L2 (L2 Regularization), ' +
                             'Dr (Dropout), ES (Early Stopping), or None (default)')
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
    
    # Process each noise level (using specific float values, never "all")
    for noise in noise_to_process:  # noise is always a float value from noise_levels
        results_dir = f'Results_{noise}'
        print(f"{'='*75}")
        print(f"\tUsing results directory: {results_dir}")
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Process each dataset (using specific dataset names, never "all")
        for dataset in datasets_to_process:  # dataset is always a string from datasets
            print(f"{'='*75}")
            print(f"\tPROCESSING NOISE LEVEL: {noise}\t\tDATASET: {dataset}")
            print(f"{'='*75}")
            
            # Prepare data for comparison
            comparison_data = []
            
            # Process each experiment
            print(f"Loading experiment data and creating epoch plots for {dataset}...\n")
            for experiment in os.listdir(results_dir):
                experiment_path = os.path.join(results_dir, experiment)
                
                # Skip if not a directory or is !Compare_All
                if not os.path.isdir(experiment_path) or experiment == '!Compare_All':
                    continue
                
                # Skip if doesn't match dataset filter
                if not experiment.lower().endswith('_' + dataset.lower()):
                    continue
                
                # Load results
                results = load_experiment_results(experiment_path)
                
                # Skip if no results found
                if not results:
                    continue
                
                # Create plots
                create_epoch_plots(results, experiment_path)
                
                # Add to comparison data
                comparison_data.append({
                    'Experiment': experiment,
                    'Test Loss': results.get('test_loss', 0),
                    'Test Accuracy': results.get('test_accuracy', 0),
                    'Computation Time': results.get('computation_time', 0)
                })
            
            # Create comprehensive comparison using specific dataset name
            print(f"Comparing all experiments for {dataset}...\n")
            compare_all_experiments(comparison_data, results_dir, dataset)  # Using specific dataset name

if __name__ == "__main__":
    main()