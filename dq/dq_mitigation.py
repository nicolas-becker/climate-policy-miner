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
    # Read Excel file from 'Mitigation' sheet, starting at row 8 (header at row 7)
    df = pd.read_excel(file_path, sheet_name='Mitigation', header=7)
    
    # Select only the columns of interest for Mitigation analysis
    columns_of_interest = [
        'Document ID',
        'Country Code',
        'Version number',
        'Category',
        'Purpose',
        'Instrument',
        'Quote',
        'A-S-I',
        'Activity type',
        'Status of measure',
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
    categorical_cols = ['Country Code', 'Version number', 'Category', 'Purpose', 'Instrument', 'A-S-I', 'Activity type', 'Status of measure']
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n   {col}:")
            value_counts = df[col].value_counts()
            if len(value_counts) > 10:
                print(value_counts.head(10))
                print(f"   ... and {len(value_counts) - 10} more values")
            else:
                print(value_counts)
    
    return df

def export_analysis_report(df, output_path):
    """
    Export analysis results to Excel with multiple sheets
    
    Args:
        df: DataFrame to analyze
        output_path: Path for output Excel file
    """
    # Create a copy with missing value markers
    df_marked = df.copy()
    
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
        
        # Duplicate rows analysis
        duplicates = df[df.duplicated(keep=False)].copy()
        if not duplicates.empty:
            # Create duplicate group IDs based on all column values
            duplicates['Duplicate_Group'] = duplicates.groupby(list(df.columns), dropna=False).ngroup() + 1
            # Sort by duplicate group and then by all columns
            duplicates = duplicates.sort_values(by=['Duplicate_Group'] + list(df.columns))
            # Reorder to put Duplicate_Group first
            cols = ['Duplicate_Group'] + [col for col in df.columns]
            duplicates = duplicates[cols]
            duplicates.to_excel(writer, sheet_name='Duplicate Rows', index=False)
        
        # Rows with any missing values
        rows_with_missing = df[df.isnull().any(axis=1)].copy()
        if not rows_with_missing.empty:
            # Add flags to this sheet too
            for col in rows_with_missing.columns:
                flag_col = f'{col}_MISSING'
                rows_with_missing[flag_col] = rows_with_missing[col].isnull().map({True: '❌', False: ''})
            rows_with_missing.to_excel(writer, sheet_name='Incomplete Rows', index=False)
        
        # Value counts for categorical columns
        categorical_cols = ['Country Code', 'Version number', 'Category', 'Purpose', 'Instrument', 'A-S-I', 'Activity type', 'Status of measure']
        for col in categorical_cols:
            if col in df.columns:
                value_counts = df[col].value_counts(dropna=False).reset_index()
                value_counts.columns = [col, 'Count']
                # Truncate sheet name to 31 characters (Excel limit)
                sheet_name = col[:31] if len(col) > 31 else col
                value_counts.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Category-Purpose relationship analysis
        if 'Category' in df.columns and 'Purpose' in df.columns:
            cat_purpose = df.groupby(['Category', 'Purpose']).size().reset_index(name='Count')
            cat_purpose = cat_purpose.sort_values(['Category', 'Count'], ascending=[True, False])
            cat_purpose.to_excel(writer, sheet_name='Category-Purpose', index=False)
        
        # Instrument by Category analysis
        if 'Category' in df.columns and 'Instrument' in df.columns:
            instrument_cat = df.groupby(['Category', 'Instrument']).size().reset_index(name='Count')
            instrument_cat = instrument_cat.sort_values(['Category', 'Count'], ascending=[True, False])
            instrument_cat.to_excel(writer, sheet_name='Instrument by Category', index=False)
        
        # A-S-I by Activity type analysis
        if 'A-S-I' in df.columns and 'Activity type' in df.columns:
            asi_activity = df.groupby(['A-S-I', 'Activity type']).size().reset_index(name='Count')
            asi_activity = asi_activity.sort_values(['A-S-I', 'Count'], ascending=[True, False])
            asi_activity.to_excel(writer, sheet_name='A-S-I by Activity', index=False)
        
        # Status by Country
        if 'Country Code' in df.columns and 'Status of measure' in df.columns:
            status_country = df.groupby(['Country Code', 'Status of measure']).size().reset_index(name='Count')
            status_country = status_country.sort_values(['Country Code', 'Count'], ascending=[True, False])
            status_country.to_excel(writer, sheet_name='Status by Country', index=False)
    
    print(f"\nReport exported to: {output_path}")

if __name__ == "__main__":
    # Specify your Excel file path
    input_file = "dq/excel/NDC-Database-Analysis_current_NEW.xlsx"
    output_file = "dq/analysis_report_mitigation.xlsx"
    
    # Run analysis
    data = analyze_excel(input_file)
    
    # Export report
    export_analysis_report(data, output_file)