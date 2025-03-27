import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def compare_all_experiments():
    """
    Create a comprehensive comparison of all experiments
    """
    results_dir = 'Results'
    experiments = [d for d in os.listdir(results_dir) 
                   if os.path.isdir(os.path.join(results_dir, d))]
    
    # Prepare data for comparison
    comparison_data = []
    for experiment in experiments:
        exp_path = os.path.join(results_dir, experiment)
        results = load_experiment_results(exp_path)
        
        comparison_data.append({
            'Experiment': experiment,
            'Test Loss': results.get('test_loss', 0),
            'Test Accuracy': results.get('test_accuracy', 0),
            'Computation Time': results.get('computation_time', 0)
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Plotting
    plt.figure(figsize=(10, 15))
    
    # Test Loss Comparison
    plt.subplot(3, 1, 1)
    bars_loss = sns.barplot(x='Experiment', y='Test Loss', data=comparison_df)
    plt.title('Test Loss Comparison')
    plt.xticks(rotation=45, ha='right')
    
    # Add value annotations for Test Loss
    for bar in bars_loss.patches:
        height = bar.get_height()
        bars_loss.text(bar.get_x() + bar.get_width()/2., height, 
                       f'{height:.4f}', 
                       ha='center', va='bottom')
    
    # Test Accuracy Comparison
    plt.subplot(3, 1, 2)
    bars_accuracy = sns.barplot(x='Experiment', y='Test Accuracy', data=comparison_df)
    plt.title('Test Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')
    
    # Add value annotations for Test Accuracy
    for bar in bars_accuracy.patches:
        height = bar.get_height()
        bars_accuracy.text(bar.get_x() + bar.get_width()/2., height, 
                           f'{height:.4f}', 
                           ha='center', va='bottom')
    
    # Computation Time Comparison
    plt.subplot(3, 1, 3)
    bars_time = sns.barplot(x='Experiment', y='Computation Time', data=comparison_df)
    plt.title('Computation Time Comparison')
    plt.xticks(rotation=45, ha='right')
    
    # Add value annotations for Computation Time
    for bar in bars_time.patches:
        height = bar.get_height()
        bars_time.text(bar.get_x() + bar.get_width()/2., height, 
                       f'{height:.4f}', 
                       ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'experiments_comparison.png'))
    plt.close()
    
    # Save comparison data
    comparison_df.to_csv(os.path.join(results_dir, 'experiments_comparison.csv'), index=False)

def main():
    """
    Analyze results for all experiments
    """
    results_dir = 'Results'
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Process each experiment
    for experiment in os.listdir(results_dir):
        experiment_path = os.path.join(results_dir, experiment)
        
        # Skip if not a directory
        if not os.path.isdir(experiment_path):
            continue
        
        # Load results
        results = load_experiment_results(experiment_path)
        
        # Skip if no results found
        if not results:
            continue
        
        # Create plots
        create_epoch_plots(results, experiment_path)
    
    # Create comprehensive comparison
    compare_all_experiments()

if __name__ == "__main__":
    main()