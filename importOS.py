import os
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

def explore_dataset(root_dir):
    # List all directories
    directories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    directories.sort()  # Sort directories alphabetically
    print(f"Found {len(directories)} directories: {directories}")
    
    # Examine each directory
    for directory in directories:
        dir_path = os.path.join(root_dir, directory)
        files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
        print(f"\nDirectory: {directory} contains {len(files)} CSV files")
        
        # Look at the first CSV file to understand structure
        if files:
            first_file_path = os.path.join(dir_path, files[0])
            df = pd.read_csv(first_file_path)
            print(f"Sample file: {files[0]}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print("First few rows:")
            print(df.head(3))


def check_missing_values(dataset_path):
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                if df.isnull().values.any():
                    print(f"Missing values found in file: {file}")
                    print(df.isnull().sum())
                else:
                    print(f"No missing values in file: {file}")

def analyze_column_statistics(dataset_path):
    all_data = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                all_data.append(df)

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)

    # Analyze input columns (2nd, 3rd, 4th)
    input_columns = combined_data.iloc[:, 1:4]
    print("Input Columns Statistics:")
    print(input_columns.describe())

    # Analyze label column (5th)
    label_column = combined_data.iloc[:, 4]
    print("\nLabel Column Distribution:")
    print(label_column.value_counts())



def visualize_data_distribution(dataset_path):
    all_data = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                all_data.append(df)

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)

    # Plot histograms for input columns
    input_columns = combined_data.iloc[:, 1:4]
    input_columns.columns = ['Feature 1', 'Feature 2', 'Feature 3']
    input_columns.hist(bins=20, figsize=(10, 6))
    plt.suptitle("Input Features Distribution")
    plt.show()

    # Plot label distribution
    label_column = combined_data.iloc[:, 4]
    sns.countplot(x=label_column)
    plt.title("Label Distribution")
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.show()
        
# Replace with your dataset directory path
def check_class_imbalance(dataset_path):
    all_data = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                all_data.append(df)

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)

    # Analyze label column
    label_column = combined_data.iloc[:, 4]
    label_counts = label_column.value_counts()
    print("Label Counts:")
    print(label_counts)

    # Check imbalance ratio
    imbalance_ratio = label_counts.max() / label_counts.min()
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")

def check_outliers(dataset_path):
    all_data = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                all_data.append(df)

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)

    # Plot boxplots for input columns
    input_columns = combined_data.iloc[:, 1:4]
    input_columns.columns = ['Feature 1', 'Feature 2', 'Feature 3']
    input_columns.boxplot(figsize=(10, 6))
    plt.title("Boxplot of Input Features")
    plt.show()

def analyze_correlation(dataset_path):
    all_data = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                all_data.append(df)

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)

    # Compute correlation matrix
    input_columns = combined_data.iloc[:, 1:4]
    correlation_matrix = input_columns.corr()

    # Plot heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix of Input Features")
    plt.show()



def find_files_with_hash(dataset_path):
    files_with_hash = []
    
    # Iterate through all CSV files in the dataset directory
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                # Load the CSV file into a DataFrame
                try:
                    df = pd.read_csv(file_path)
                    
                    # Check if the 5th column contains '#'
                    if '#' in df.iloc[:, 4].values:
                        files_with_hash.append(file)
                
                except Exception as e:
                    print(f"Error reading file {file}: {e}")
    
    # Print results
    if files_with_hash:
        print("Files containing '#' in the label column:")
        for f in files_with_hash:
            print(f)
    else:
        print("No files contain '#' in the label column.")


# Function to verify dataset integrity
def verify_dataset_integrity(dataset_path):
    label_issues = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    # Check if the label column (5th column) contains only one unique value
                    unique_labels = df.iloc[:, 4].unique()
                    if len(unique_labels) == 1:
                        label_issues.append((file, unique_labels))
                except Exception as e:
                    print(f"Error reading file {file}: {e}")

    return label_issues



# Function to check encoding issues
def check_encoding_issues(dataset_path):
    encoding_issues = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    # Attempt to open the file with UTF-8 encoding
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.read()
                except UnicodeDecodeError:
                    encoding_issues.append(file)

    return encoding_issues



# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # DEFINE DATASET PATH # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 

dataset_path = "eel4810-dataset"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # FUNCTIONS FOR DATA ANALYSIS # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # UNCOMMENT THE FUNCTION CALLS AS NEEDED  # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Replace with your dataset directory path
## explore_dataset(dataset_path)

# Replace with your dataset directory path
## check_missing_values(dataset_path)

# Replace with your dataset directory path
## analyze_column_statistics(dataset_path)

# Replace with your dataset directory path
## visualize_data_distribution(dataset_path)

# Replace with your dataset directory path
## check_class_imbalance(dataset_path)

# Replace with your dataset directory path
## check_outliers(dataset_path)

# Replace with your dataset directory path
## analyze_correlation(dataset_path)

# Replace with your dataset directory path
## find_files_with_hash(dataset_path)

# Verify dataset integrity
## label_issues = verify_dataset_integrity(dataset_path)
## if label_issues:
##     print("Files with single unique labels:")
##     for file, labels in label_issues:
##         print(f"{file}: {labels}")
## else:
##     print("No issues found with dataset integrity.")


# Check for encoding issues
## encoding_issues = check_encoding_issues(dataset_path)
## if encoding_issues:
##     print("Files with encoding issues:")
##     for file in encoding_issues:
##         print(file)
## else:
##     print("No encoding issues found.")