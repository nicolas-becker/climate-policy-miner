import pandas as pd
import openpyxl
from pathlib import Path
import re

def analyze_excel(file_path):
    """
    Perform basic analysis on an Excel file
    
    Args:
        file_path: Path to the Excel file
    """
    # Read Excel file from 'Targets' sheet, starting at row 8 (header at row 7)
    df = pd.read_excel(file_path, sheet_name='Targets', header=7)
    
    # Select only the columns of interest
    columns_of_interest = [
        'Document ID',
        'Country Code',
        'Version number',
        'Target area',
        'Target scope',
        'GHG target?',
        'Target type',
        'Conditionality',
        'Target Year',
        'Content',
        'Page Number'
    ]
    
    # Filter to only include these columns (handles case variations)
    available_columns = [col for col in columns_of_interest if col in df.columns]
    df = df[available_columns]
    
    print(f"Analysis of: {file_path}")
    print("=" * 50)
    
    # Basic information
    print("\n1. Dataset Overview:")
    print(f"   - Rows: {len(df)}")
    print(f"   - Columns: {len(df.columns)}")
    print(f"   - Column names: {list(df.columns)}")
    
    # Data types
    print("\n2. Data Types:")
    print(df.dtypes)
    
    # Missing values
    print("\n3. Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    # Descriptive statistics
    print("\n4. Descriptive Statistics:")
    print(df.describe(include='all'))
    
    # Duplicate rows
    print("\n5. Duplicate Rows:")
    duplicates = df.duplicated().sum()
    print(f"   - Number of duplicates: {duplicates}")
    
    # Value counts for categorical columns
    print("\n6. Value Counts for Key Columns:")
    categorical_cols = ['Version number', 'Target area', 'Target scope', 'Target type', 'Conditionality']
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n   {col}:")
            print(df[col].value_counts())
    
    # Check if Target Year is in Content
    print("\n7. Target Year Validation:")
    year_in_content = check_year_in_content(df)
    print(f"   - Rows where Target Year is NOT in Content: {year_in_content['missing'].sum()}")
    print(f"   - Percentage: {(year_in_content['missing'].sum() / len(df)) * 100:.2f}%")
    print(f"   - Rows with suggested alternative years: {(year_in_content['suggested_years'].notna()).sum()}")
    
    return df

def extract_years_from_text(text, future_only=False):
    """
    Extract all 4-digit years from text
    
    Args:
        text: String to extract years from
        future_only: If True, only return years >= 2015
    
    Returns:
        List of years found
    """
    if pd.isna(text):
        return []
    
    # Match complete 4-digit years (2015-2099)
    years = re.findall(r'\b(20\d{2})\b', str(text))
    
    if future_only:
        # Filter to only include years from 2015 onwards
        years = [year for year in years if int(year) >= 2015]
    
    return years

def check_year_in_content(df):
    """
    Check if Target Year appears in the Content column
    
    Args:
        df: DataFrame with 'Target Year' and 'Content' columns
    
    Returns:
        DataFrame with validation results
    """
    results = pd.DataFrame()
    
    if 'Target Year' in df.columns and 'Content' in df.columns:
        def year_exists_in_content(row):
            if pd.isna(row['Target Year']) or pd.isna(row['Content']):
                return None, None  # Can't validate if either is missing
            
            # Convert Target Year to string and extract all 4-digit years
            target_year_str = str(row['Target Year'])
            content_str = str(row['Content'])
            
            # Extract years from target year (handles ranges like "2030", "2030-2035", etc.)
            years_in_target = extract_years_from_text(target_year_str)
            
            # Check if any of these years appear in content
            for year in years_in_target:
                if year in content_str:
                    return True, None
            
            # If year not found, extract future years from content as suggestions
            years_in_content = extract_years_from_text(content_str, future_only=True)
            suggested_years = ', '.join(sorted(set(years_in_content))) if years_in_content else None
            
            return False, suggested_years
        
        validation_results = df.apply(year_exists_in_content, axis=1, result_type='expand')
        results['year_in_content'] = validation_results[0]
        results['suggested_years'] = validation_results[1]
        results['missing'] = results['year_in_content'] == False
        results['cannot_validate'] = results['year_in_content'].isna()
    
    return results

def export_analysis_report(df, output_path):
    """
    Export analysis results to Excel with multiple sheets
    
    Args:
        df: DataFrame to analyze
        output_path: Path for output Excel file
    """
    # Create a copy with missing value markers
    df_marked = df.copy()
    
    # Check year in content
    year_check = check_year_in_content(df)
    df_marked['Year_In_Content'] = year_check['year_in_content'].map({
        True: '✓', 
        False: '❌ YEAR NOT FOUND', 
        None: 'N/A'
    })
    df_marked['Suggested_Years'] = year_check['suggested_years']
    
    # Add flag columns for missing values
    for col in df.columns:
        flag_col = f'{col}_MISSING'
        df_marked[flag_col] = df[col].isnull().map({True: '❌ MISSING', False: ''})
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Original data with missing flags
        df_marked.to_excel(writer, sheet_name='Data with Flags', index=False)
        
        # Original data only
        df.to_excel(writer, sheet_name='Data', index=False)
        
        # Summary statistics
        df.describe(include='all').to_excel(writer, sheet_name='Statistics')
        
        # Missing values summary
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': missing.values,
            'Percentage': missing_pct
        })
        missing_df.to_excel(writer, sheet_name='Missing Values', index=False)
        
        # Year validation issues with suggestions
        year_issues = df_marked[df_marked['Year_In_Content'] == '❌ YEAR NOT FOUND'].copy()
        if not year_issues.empty:
            # Select only original columns plus Year_In_Content and Suggested_Years
            # Drop all the _MISSING flag columns
            cols_to_keep = [col for col in year_issues.columns if not col.endswith('_MISSING')]
            year_issues = year_issues[cols_to_keep]
            
            # Reorder to show suggested years prominently after Target Year
            if 'Suggested_Years' in cols_to_keep and 'Target Year' in cols_to_keep:
                cols_to_keep.remove('Suggested_Years')
                if 'Year_In_Content' in cols_to_keep:
                    cols_to_keep.remove('Year_In_Content')
                target_year_idx = cols_to_keep.index('Target Year')
                cols_to_keep.insert(target_year_idx + 1, 'Suggested_Years')
                cols_to_keep.insert(target_year_idx + 2, 'Year_In_Content')
            
            year_issues = year_issues[cols_to_keep]
            year_issues.to_excel(writer, sheet_name='Year Not In Content', index=False)
        
        # Rows with any missing values
        rows_with_missing = df[df.isnull().any(axis=1)].copy()
        if not rows_with_missing.empty:
            # Add flags to this sheet too
            for col in rows_with_missing.columns:
                flag_col = f'{col}_MISSING'
                rows_with_missing[flag_col] = rows_with_missing[col].isnull().map({True: '❌', False: ''})
            rows_with_missing.to_excel(writer, sheet_name='Incomplete Rows', index=False)
        
        # Value counts for categorical columns
        categorical_cols = ['Version number', 'Target area', 'Target scope', 'Target type', 'Conditionality']
        for col in categorical_cols:
            if col in df.columns:
                value_counts = df[col].value_counts(dropna=False).reset_index()
                value_counts.columns = [col, 'Count']
                value_counts.to_excel(writer, sheet_name=f'{col[:25]}', index=False)
    
    print(f"\nReport exported to: {output_path}")

if __name__ == "__main__":
    # Specify your Excel file path
    input_file = "dq/excel/NDC-Database-Analysis_current_NEW.xlsx"
    output_file = "dq/analysis_report_targets.xlsx"
    
    # Run analysis
    data = analyze_excel(input_file)
    
    # Export report
    export_analysis_report(data, output_file)