import streamlit as st
import pandas as pd
from io import BytesIO, StringIO

# Function to add a serial number column to DataFrame
def add_serial_number_column(df):
    # Remove duplicates of 'sr. no.' if present
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Add serial number column
    if 'sr. no.' not in df.columns:
        df.insert(0, 'sr. no.', range(1, len(df) + 1))
    
    return df

# Function to display DataFrame with additional statistics
def display_dataframe_with_stats(df, name, sum_column):
    st.header(name)
    st.dataframe(add_serial_number_column(df))
    
    row_count = len(df)
    total_sum = df[sum_column].sum() if sum_column in df.columns else 0
    
    st.write(f"**{name} - Total Rows:** {row_count}")
    st.write(f"**{name} - Sum of {sum_column}:** {total_sum:.2f}")

def add_empty_line(input_content, target_line):
    output = StringIO()
    for line in input_content.split('\n'):
        output.write(line + '\n')
        if line.strip() == target_line.strip():
            output.write('\n')
    return output.getvalue()

def add_line_breaker_to_content(content):
    sections = content.split('^PART-I - Details of Tax Deducted at Source^')
    
    if len(sections) < 2:
        raise ValueError("Expected header not found in the file")

    header_section = sections[0]
    data_section = sections[1]

    lines = data_section.strip().split('\n')
    modified_lines = []
    header_found = False

    for line in lines:
        if not header_found and "Sr. No." in line:
            modified_lines.append(line)
            modified_lines.append(' ' * 1)
            header_found = True
        else:
            modified_lines.append(line)

    modified_content = header_section + '^PART-I - Details of Tax Deducted at Source^' + '\n'.join(modified_lines)
    return modified_content

def read_data_from_content(content):
    sections = content.split('^PART-I - Details of Tax Deducted at Source^')[1].split('\n\n')

    all_data = []
    header = None

    for section in sections:
        lines = section.strip().split('\n')
        if not lines:
            continue

        deductor_info = lines[0].split('^')
        if len(deductor_info) < 3:
            continue

        deductor_number = deductor_info[0]
        deductor_name = deductor_info[1]
        deductor_tan = deductor_info[2]

        for line in lines[1:]:
            if line.strip() == '':
                continue
            
            cols = [col.strip() for col in line.split('^') if col.strip()]
            if not header and "Sr. No." in cols:
                header = cols
            elif header and cols and cols[0].isdigit() and len(cols) == len(header):
                all_data.append([deductor_number, deductor_name, deductor_tan] + cols)

    if not header:
        raise ValueError("Header not found in the file")

    full_header = ['Deductor Number', 'Name of Deductor', 'TAN of Deductor'] + header
    return full_header, all_data

def create_tds_dataframe(header, data):
    df = pd.DataFrame(data, columns=header)
    numeric_columns = ['Amount Paid / Credited(Rs.)', 'Tax Deducted(Rs.)', 'TDS Deposited(Rs.)']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def process_tds_file(uploaded_file):
    """Reads and processes the TDS file to extract data and create a DataFrame."""
    content = uploaded_file.getvalue().decode("utf-8")
    target_line = "Sr. No.^Name of Deductor^TAN of Deductor^^^^^Total Amount Paid / Credited(Rs.)^Total Tax Deducted(Rs.)^Total TDS Deposited(Rs.)"
    content_with_empty_line = add_empty_line(content, target_line)
    modified_content = add_line_breaker_to_content(content_with_empty_line)
    header, data = read_data_from_content(modified_content)
    df = create_tds_dataframe(header, data)
    df = df.drop(columns=['Deductor Number', 'Sr. No.'], errors='ignore')

    aggregated_tds_df = df.groupby(['Name of Deductor', 'TAN of Deductor']).agg({
        'Amount Paid / Credited(Rs.)': 'sum',
        'Tax Deducted(Rs.)': 'sum',
        'TDS Deposited(Rs.)': 'sum'
    }).reset_index()
    
    return df, aggregated_tds_df

def preprocess_zoho_file(file):
    """Reads and processes the Zoho Excel file to extract and aggregate data."""
    try:
        df = pd.read_excel(file, skiprows=1)
        required_columns = ['transaction_details', 'net_amount']

        if not all(col in df.columns for col in required_columns):
            raise ValueError("Required columns not found in Zoho Book data")
        
        selected_columns = df.loc[:, required_columns].copy()
        selected_columns.rename(columns={
            'transaction_details': 'name of the deductor',
            'net_amount': 'tds of the current fin. year'
        }, inplace=True)
        selected_columns.columns = selected_columns.columns.str.strip().str.lower()
        selected_columns['name of the deductor'] = selected_columns['name of the deductor'].str.upper()
        selected_columns['tds of the current fin. year'] = pd.to_numeric(selected_columns['tds of the current fin. year'], errors='coerce')
        
        selected_columns = selected_columns.iloc[:-1]
        aggregated_df = selected_columns.groupby('name of the deductor', as_index=False)['tds of the current fin. year'].sum()
        
        return aggregated_df, selected_columns
    except Exception as e:
        st.error(f"Error processing Zoho file: {str(e)}")
        return None, None

def compare_dataframes(aggregated_tds_df, aggregated_zoho_df):
    """Compares TDS and Zoho DataFrames and returns a new DataFrame with exact matching entries."""
    matching_df = pd.merge(
        aggregated_tds_df,
        aggregated_zoho_df[['name of the deductor', 'tds of the current fin. year']],
        left_on=['Name of Deductor', 'TDS Deposited(Rs.)'],
        right_on=['name of the deductor', 'tds of the current fin. year'],
        how='inner'
    )
    matching_df = matching_df.rename(
        columns={
            'name of the deductor': 'Name of Deductor (Zoho)',
            'tds of the current fin. year': 'TDS of the Current Fin. Year (Zoho)'
        }
    )
    return matching_df

def compare_with_tolerance(aggregated_tds_df, aggregated_zoho_df, exact_matches, tolerance=10):
    """Compares TDS and Zoho DataFrames within a tolerance and excludes exact matches."""
    # Merge with tolerance and exclude exact matches
    tolerance_df = pd.merge(
        aggregated_tds_df,
        aggregated_zoho_df[['name of the deductor', 'tds of the current fin. year']],
        left_on='Name of Deductor',
        right_on='name of the deductor',
        how='inner'
    )

    # Filter within tolerance and exclude exact matches
    tolerance_df = tolerance_df[
        (abs(tolerance_df['TDS Deposited(Rs.)'] - tolerance_df['tds of the current fin. year']) <= tolerance) &
        (~tolerance_df['Name of Deductor'].isin(exact_matches['Name of Deductor'])) &
        (~tolerance_df['TDS Deposited(Rs.)'].isin(exact_matches['TDS Deposited(Rs.)']))
    ]

    tolerance_df = tolerance_df.rename(
        columns={
            'name of the deductor': 'Name of Deductor (Zoho)',
            'tds of the current fin. year': 'TDS of the Current Fin. Year (Zoho)'
        }
    )
    return tolerance_df

def get_unmatched_entries(aggregated_tds_df, aggregated_zoho_df, exact_matching_df, tolerance_matching_df):
    """Finds unmatched entries between TDS and Zoho DataFrames, excluding exact and tolerance matches."""
    # Combine exact and tolerance matches to exclude from unmatched
    combined_matches = pd.concat([exact_matching_df[['Name of Deductor', 'TDS Deposited(Rs.)']],
                                  tolerance_matching_df[['Name of Deductor', 'TDS Deposited(Rs.)']]]).drop_duplicates()

    # Unmatched in TDS
    unmatched_tds = aggregated_tds_df[~aggregated_tds_df.set_index(['Name of Deductor', 'TDS Deposited(Rs.)']).index.isin(combined_matches.set_index(['Name of Deductor', 'TDS Deposited(Rs.)']).index)]

    # Combine exact and tolerance matches to exclude from unmatched
    combined_matches_zoho = pd.concat([exact_matching_df[['Name of Deductor (Zoho)', 'TDS of the Current Fin. Year (Zoho)']],
                                       tolerance_matching_df[['Name of Deductor (Zoho)', 'TDS of the Current Fin. Year (Zoho)']]]).drop_duplicates()

    # Unmatched in Zoho
    unmatched_zoho = aggregated_zoho_df[~aggregated_zoho_df.set_index(['name of the deductor', 'tds of the current fin. year']).index.isin(combined_matches_zoho.set_index(['Name of Deductor (Zoho)', 'TDS of the Current Fin. Year (Zoho)']).index)]

    return unmatched_tds, unmatched_zoho

def get_individual_unmatched_entries(df, unmatched_df, key_col, sum_col):
    """Gets individual entries for unmatched deductors."""
    unmatched_deductors = unmatched_df[key_col].unique()
    individual_unmatched_df = df[df[key_col].isin(unmatched_deductors)]
    return individual_unmatched_df

@st.cache_data
def convert_df_to_excel(df):
    """Converts DataFrame to Excel format for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def match_individual_entries(unmatched_tds, unmatched_zoho, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho):
    """Matches individual entries between TDS and Zoho DataFrames and returns a new DataFrame with matched entries."""
    matched_indices_tds = set()
    matched_indices_zoho = set()
    
    matched_entries = []

    # Iterate over TDS DataFrame
    for idx_tds, row_tds in unmatched_tds.iterrows():
        # Skip if TDS entry already matched
        if idx_tds in matched_indices_tds:
            continue
        
        # Find a matching entry in Zoho DataFrame
        for idx_zoho, row_zoho in unmatched_zoho.iterrows():
            # Skip if Zoho entry already matched
            if idx_zoho in matched_indices_zoho:
                continue
            
            # Check if both key and sum columns match
            if (row_tds[key_col_tds] == row_zoho[key_col_zoho] and 
                row_tds[sum_col_tds] == row_zoho[sum_col_zoho]):
                
                # Add to matched entries list
                matched_entries.append({
                    key_col_tds: row_tds[key_col_tds],
                    sum_col_tds: row_tds[sum_col_tds],
                    key_col_zoho: row_zoho[key_col_zoho],
                    sum_col_zoho: row_zoho[sum_col_zoho]
                })
                
                # Mark these indices as matched
                matched_indices_tds.add(idx_tds)
                matched_indices_zoho.add(idx_zoho)
                
                # Break after finding the first match
                break

    # Convert matched entries to DataFrame
    matched_df = pd.DataFrame(matched_entries)
    return matched_df

def match_individual_entries_with_tolerance(unmatched_tds, unmatched_zoho, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho, tolerance):
    """Matches individual entries between TDS and Zoho DataFrames within a tolerance and returns a new DataFrame with matched entries."""
    matched_indices_tds = set()
    matched_indices_zoho = set()
    
    matched_entries = []

    # Iterate over TDS DataFrame
    for idx_tds, row_tds in unmatched_tds.iterrows():
        # Skip if TDS entry already matched
        if idx_tds in matched_indices_tds:
            continue
        
        # Find a matching entry in Zoho DataFrame within the tolerance range
        for idx_zoho, row_zoho in unmatched_zoho.iterrows():
            # Skip if Zoho entry already matched
            if idx_zoho in matched_indices_zoho:
                continue
            
            # Check if both key columns match and sum columns are within the tolerance range
            if (row_tds[key_col_tds] == row_zoho[key_col_zoho] and 
                abs(row_tds[sum_col_tds] - row_zoho[sum_col_zoho]) <= tolerance):
                
                # Add to matched entries list
                matched_entries.append({
                    key_col_tds: row_tds[key_col_tds],
                    sum_col_tds: row_tds[sum_col_tds],
                    key_col_zoho: row_zoho[key_col_zoho],
                    sum_col_zoho: row_zoho[sum_col_zoho]
                })
                
                # Mark these indices as matched
                matched_indices_tds.add(idx_tds)
                matched_indices_zoho.add(idx_zoho)
                
                # Break after finding the first match
                break

    # Convert matched entries to DataFrame
    matched_df = pd.DataFrame(matched_entries)
    return matched_df

def get_remaining_unmatched_entries(individual_unmatched_tds, individual_unmatched_zoho, matched_df, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho):
    """Finds the remaining unmatched individual entries after removing only the exact matched ones."""

    # Create copies of the unmatched DataFrames to avoid modifying the originals
    remaining_unmatched_tds = individual_unmatched_tds.copy()
    remaining_unmatched_zoho = individual_unmatched_zoho.copy()

    # Convert matched_df to a list of dictionaries for efficient row-wise processing
    matched_rows = matched_df.to_dict('records')

    # Iterate through each matched row to remove only the exact matches
    for matched_row in matched_rows:
        # Find the first exact match in the TDS DataFrame
        tds_matches = remaining_unmatched_tds[
            (remaining_unmatched_tds[key_col_tds] == matched_row[key_col_tds]) &
            (remaining_unmatched_tds[sum_col_tds] == matched_row[sum_col_tds])
        ]

        # Remove only the first matched row in TDS if it exists
        if not tds_matches.empty:
            first_match_idx = tds_matches.index[0]
            remaining_unmatched_tds = remaining_unmatched_tds.drop(index=first_match_idx)

        # Find the first exact match in the Zoho DataFrame
        zoho_matches = remaining_unmatched_zoho[
            (remaining_unmatched_zoho[key_col_zoho] == matched_row[key_col_zoho]) &
            (remaining_unmatched_zoho[sum_col_zoho] == matched_row[sum_col_zoho])
        ]

        # Remove only the first matched row in Zoho if it exists
        if not zoho_matches.empty:
            first_match_idx = zoho_matches.index[0]
            remaining_unmatched_zoho = remaining_unmatched_zoho.drop(index=first_match_idx)

    return remaining_unmatched_tds, remaining_unmatched_zoho

def get_remaining_unmatched_entries_with_tolerance(individual_unmatched_tds, individual_unmatched_zoho, matched_df, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho):
    """Finds the remaining unmatched individual entries after removing only the matched ones within the tolerance range."""

    # Create copies of the unmatched DataFrames to avoid modifying the originals
    remaining_unmatched_tds = individual_unmatched_tds.copy()
    remaining_unmatched_zoho = individual_unmatched_zoho.copy()

    # Convert matched_df to a list of dictionaries for efficient row-wise processing
    matched_rows = matched_df.to_dict('records')

    # Iterate through each matched row to remove only the exact matches
    for matched_row in matched_rows:
        # Find the first exact match in the TDS DataFrame
        tds_matches = remaining_unmatched_tds[
            (remaining_unmatched_tds[key_col_tds] == matched_row[key_col_tds]) &
            (remaining_unmatched_tds[sum_col_tds] == matched_row[sum_col_tds])
        ]

        # Remove only the first matched row in TDS if it exists
        if not tds_matches.empty:
            first_match_idx = tds_matches.index[0]
            remaining_unmatched_tds = remaining_unmatched_tds.drop(index=first_match_idx)

        # Find the first exact match in the Zoho DataFrame
        zoho_matches = remaining_unmatched_zoho[
            (remaining_unmatched_zoho[key_col_zoho] == matched_row[key_col_zoho]) &
            (remaining_unmatched_zoho[sum_col_zoho] == matched_row[sum_col_zoho])
        ]

        # Remove only the first matched row in Zoho if it exists
        if not zoho_matches.empty:
            first_match_idx = zoho_matches.index[0]
            remaining_unmatched_zoho = remaining_unmatched_zoho.drop(index=first_match_idx)

    return remaining_unmatched_tds, remaining_unmatched_zoho

def extract_first_three_words(name):
    """Extracts the first three words from the given string."""
    if isinstance(name, str):
        return ' '.join(name.split()[:3])
    return ''

def match_individual_entries_based_on_three_words(unmatched_tds, unmatched_zoho, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho):
    """Matches individual entries between TDS and Zoho DataFrames based on the first three words in the deductor name."""
    
    # Extract the first three words from the deductor names
    unmatched_tds['Three Words Deductor (TDS)'] = unmatched_tds[key_col_tds].apply(extract_first_three_words)
    unmatched_zoho['Three Words Deductor (Zoho)'] = unmatched_zoho[key_col_zoho].apply(extract_first_three_words)
    
    matched_indices_tds = set()
    matched_indices_zoho = set()
    
    matched_entries = []

    # Iterate over TDS DataFrame
    for idx_tds, row_tds in unmatched_tds.iterrows():
        # Skip if TDS entry already matched
        if idx_tds in matched_indices_tds:
            continue
        
        # Find a matching entry in Zoho DataFrame
        for idx_zoho, row_zoho in unmatched_zoho.iterrows():
            # Skip if Zoho entry already matched
            if idx_zoho in matched_indices_zoho:
                continue
            
            # Check if both key columns match on first three words and sum columns match
            if (row_tds['Three Words Deductor (TDS)'] == row_zoho['Three Words Deductor (Zoho)'] and 
                row_tds[sum_col_tds] == row_zoho[sum_col_zoho]):
                
                # Add to matched entries list
                matched_entries.append({
                    key_col_tds: row_tds[key_col_tds],
                    sum_col_tds: row_tds[sum_col_tds],
                    key_col_zoho: row_zoho[key_col_zoho],
                    sum_col_zoho: row_zoho[sum_col_zoho]
                })
                
                # Mark these indices as matched
                matched_indices_tds.add(idx_tds)
                matched_indices_zoho.add(idx_zoho)
                
                # Break after finding the first match
                break

    # Convert matched entries to DataFrame
    matched_df = pd.DataFrame(matched_entries)
    
    # Drop the temporary columns used for matching
    unmatched_tds.drop(columns=['Three Words Deductor (TDS)'], inplace=True)
    unmatched_zoho.drop(columns=['Three Words Deductor (Zoho)'], inplace=True)
    
    return matched_df

def get_remaining_unmatched_entries_after_three_words(individual_unmatched_tds, individual_unmatched_zoho, matched_df, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho):
    """Finds the remaining unmatched individual entries after removing the matched ones based on the first three words in the deductor name."""

    # Create copies of the unmatched DataFrames to avoid modifying the originals
    remaining_unmatched_tds = individual_unmatched_tds.copy()
    remaining_unmatched_zoho = individual_unmatched_zoho.copy()

    # Convert matched_df to a list of dictionaries for efficient row-wise processing
    matched_rows = matched_df.to_dict('records')

    # Iterate through each matched row to remove only the exact matches
    for matched_row in matched_rows:
        # Find the first exact match in the TDS DataFrame
        tds_matches = remaining_unmatched_tds[
            (remaining_unmatched_tds[key_col_tds] == matched_row[key_col_tds]) &
            (remaining_unmatched_tds[sum_col_tds] == matched_row[sum_col_tds])
        ]

        # Remove only the first matched row in TDS if it exists
        if not tds_matches.empty:
            first_match_idx = tds_matches.index[0]
            remaining_unmatched_tds = remaining_unmatched_tds.drop(index=first_match_idx)

        # Find the first exact match in the Zoho DataFrame
        zoho_matches = remaining_unmatched_zoho[
            (remaining_unmatched_zoho[key_col_zoho] == matched_row[key_col_zoho]) &
            (remaining_unmatched_zoho[sum_col_zoho] == matched_row[sum_col_zoho])
        ]

        # Remove only the first matched row in Zoho if it exists
        if not zoho_matches.empty:
            first_match_idx = zoho_matches.index[0]
            remaining_unmatched_zoho = remaining_unmatched_zoho.drop(index=first_match_idx)

    return remaining_unmatched_tds, remaining_unmatched_zoho

def match_individual_entries_with_tolerance_based_on_three_words(unmatched_tds, unmatched_zoho, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho, tolerance):
    """Matches individual entries between TDS and Zoho DataFrames within a tolerance based on the first three words in the deductor name."""
    
    # Extract the first three words from the deductor names
    unmatched_tds['Three Words Deductor (TDS)'] = unmatched_tds[key_col_tds].apply(extract_first_three_words)
    unmatched_zoho['Three Words Deductor (Zoho)'] = unmatched_zoho[key_col_zoho].apply(extract_first_three_words)
    
    matched_indices_tds = set()
    matched_indices_zoho = set()
    
    matched_entries = []

    # Iterate over TDS DataFrame
    for idx_tds, row_tds in unmatched_tds.iterrows():
        # Skip if TDS entry already matched
        if idx_tds in matched_indices_tds:
            continue
        
        # Find a matching entry in Zoho DataFrame within the tolerance range
        for idx_zoho, row_zoho in unmatched_zoho.iterrows():
            # Skip if Zoho entry already matched
            if idx_zoho in matched_indices_zoho:
                continue
            
            # Check if both key columns match on first three words and sum columns are within the tolerance range
            if (row_tds['Three Words Deductor (TDS)'] == row_zoho['Three Words Deductor (Zoho)'] and 
                abs(row_tds[sum_col_tds] - row_zoho[sum_col_zoho]) <= tolerance):
                
                # Add to matched entries list
                matched_entries.append({
                    key_col_tds: row_tds[key_col_tds],
                    sum_col_tds: row_tds[sum_col_tds],
                    key_col_zoho: row_zoho[key_col_zoho],
                    sum_col_zoho: row_zoho[sum_col_zoho]
                })
                
                # Mark these indices as matched
                matched_indices_tds.add(idx_tds)
                matched_indices_zoho.add(idx_zoho)
                
                # Break after finding the first match
                break

    # Convert matched entries to DataFrame
    matched_df = pd.DataFrame(matched_entries)
    
    # Drop the temporary columns used for matching
    unmatched_tds.drop(columns=['Three Words Deductor (TDS)'], inplace=True)
    unmatched_zoho.drop(columns=['Three Words Deductor (Zoho)'], inplace=True)
    
    return matched_df

def get_remaining_unmatched_entries_after_tolerance_three_words(individual_unmatched_tds, individual_unmatched_zoho, matched_df, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho):
    """Finds the remaining unmatched individual entries after removing the matched ones based on tolerance of the first three words in the deductor name."""

    # Create copies of the unmatched DataFrames to avoid modifying the originals
    remaining_unmatched_tds = individual_unmatched_tds.copy()
    remaining_unmatched_zoho = individual_unmatched_zoho.copy()

    # Convert matched_df to a list of dictionaries for efficient row-wise processing
    matched_rows = matched_df.to_dict('records')

    # Iterate through each matched row to remove only the exact matches
    for matched_row in matched_rows:
        # Find the first exact match in the TDS DataFrame
        tds_matches = remaining_unmatched_tds[
            (remaining_unmatched_tds[key_col_tds] == matched_row[key_col_tds]) &
            (remaining_unmatched_tds[sum_col_tds] == matched_row[sum_col_tds])
        ]

        # Remove only the first matched row in TDS if it exists
        if not tds_matches.empty:
            first_match_idx = tds_matches.index[0]
            remaining_unmatched_tds = remaining_unmatched_tds.drop(index=first_match_idx)

        # Find the first exact match in the Zoho DataFrame
        zoho_matches = remaining_unmatched_zoho[
            (remaining_unmatched_zoho[key_col_zoho] == matched_row[key_col_zoho]) &
            (remaining_unmatched_zoho[sum_col_zoho] == matched_row[sum_col_zoho])
        ]

        # Remove only the first matched row in Zoho if it exists
        if not zoho_matches.empty:
            first_match_idx = zoho_matches.index[0]
            remaining_unmatched_zoho = remaining_unmatched_zoho.drop(index=first_match_idx)

    return remaining_unmatched_tds, remaining_unmatched_zoho


def main():
    """Main function to handle the Streamlit app logic."""
    st.title("TDS Reconciliation Tool")
    st.sidebar.title("File Uploads")
    
    # About the Tool
    st.markdown("""
    ### About the Tool
    **Reconciliation Purpose**  
    This tool is designed to reconcile TDS data between your books of accounts (TDS Ledger) and Form 26AS.

    **Objective**  
    The goal is to minimize unmatched TDS entries as much as possible. The tool will provide reconciliation items that can be matched one-on-one with the TDS ledger.

    **Current Functionality**  
    At present, this tool summarizes TDS data on a totality basis.

    **Output**  
    The tool will generate the following columns in sequential order:

    1. Deductor Number  
    2. Name of Deductor  
    3. TAN of Deductor  
    4. Sr. No.  
    5. Section  
    6. Transaction Date  
    7. Status of Booking  
    8. Date of Booking  
    9. Remarks  
    10. Total Amount Paid / Credited (Rs.)  
    11. Total Tax Deducted (Rs.)  
    12. Total TDS Deposited (Rs.)

    *Some columns may remain empty as the data is retrieved on a totality basis.*

    Kindly note this tool is currently under development. Please review the results carefully before relying on them.
    """)

    uploaded_file = st.sidebar.file_uploader("Upload a Text File (26AS TDS File)", type=["txt"])
    zoho_file = st.sidebar.file_uploader("Upload Zoho Books Excel File", type=["xlsx"])

    if st.sidebar.button("Submit"):
        if uploaded_file is not None:
            try:
                # Process TDS File
                df, aggregated_tds_df = process_tds_file(uploaded_file)
                display_dataframe_with_stats(df, "26AS Extracted Data", "TDS Deposited(Rs.)")
                display_dataframe_with_stats(aggregated_tds_df, "Aggregated Totals by Deductor (26AS TDS)", "TDS Deposited(Rs.)")

                # Process Zoho File if available
                if zoho_file is not None:
                    aggregated_zoho_df, selected_columns = preprocess_zoho_file(zoho_file)

                    if aggregated_zoho_df is not None and selected_columns is not None:
                        display_dataframe_with_stats(selected_columns, "Zoho Extracted Data (Individual Records)", "tds of the current fin. year")
                        display_dataframe_with_stats(aggregated_zoho_df, "Aggregated Totals by Deductor (Zoho)", "tds of the current fin. year")

                        # Exact Match Comparison
                        exact_matching_df = compare_dataframes(aggregated_tds_df, aggregated_zoho_df)
                        display_dataframe_with_stats(exact_matching_df, "Aggregate Matching of TDS and Zoho", "TDS Deposited(Rs.)")

                        # Tolerance Match Comparison
                        tolerance_matching_df = compare_with_tolerance(aggregated_tds_df, aggregated_zoho_df, exact_matching_df, tolerance=10)
                        display_dataframe_with_stats(tolerance_matching_df, "Tolerance Matching of TDS and Zoho (Within ±10)", "TDS Deposited(Rs.)")

                        # Unmatched Entries
                        unmatched_tds, unmatched_zoho = get_unmatched_entries(aggregated_tds_df, aggregated_zoho_df, exact_matching_df, tolerance_matching_df)
                        display_dataframe_with_stats(unmatched_tds, "Unmatched Entries in TDS", "TDS Deposited(Rs.)")
                        display_dataframe_with_stats(unmatched_zoho, "Unmatched Entries in Zoho", "tds of the current fin. year")

                        # Display individual unmatched deductor data
                        individual_unmatched_tds = get_individual_unmatched_entries(df, unmatched_tds, 'Name of Deductor', 'TDS Deposited(Rs.)')
                        display_dataframe_with_stats(individual_unmatched_tds, "Individual Unmatched Deductors in TDS", "TDS Deposited(Rs.)")

                        individual_unmatched_zoho = get_individual_unmatched_entries(selected_columns, unmatched_zoho, 'name of the deductor', 'tds of the current fin. year')
                        display_dataframe_with_stats(individual_unmatched_zoho, "Individual Unmatched Deductors in Zoho", "tds of the current fin. year")

                        # Perform matching on individual unmatched entries
                        individual_matched_df = match_individual_entries(
                            individual_unmatched_tds, 
                            individual_unmatched_zoho, 
                            'Name of Deductor', 
                            'name of the deductor', 
                            'TDS Deposited(Rs.)', 
                            'tds of the current fin. year'
                        )
                        display_dataframe_with_stats(individual_matched_df, "Matched Individual Unmatched Entries in TDS and Zoho", "TDS Deposited(Rs.)")

                        # Get remaining unmatched individual entries after removing matched ones
                        remaining_unmatched_tds, remaining_unmatched_zoho = get_remaining_unmatched_entries(
                            individual_unmatched_tds, 
                            individual_unmatched_zoho, 
                            individual_matched_df, 
                            'Name of Deductor', 
                            'name of the deductor', 
                            'TDS Deposited(Rs.)', 
                            'tds of the current fin. year'
                        )

                        # Display remaining unmatched individual entries
                        display_dataframe_with_stats(remaining_unmatched_tds, "Remaining Unmatched Individual Entries in TDS", "TDS Deposited(Rs.)")
                        display_dataframe_with_stats(remaining_unmatched_zoho, "Remaining Unmatched Individual Entries in Zoho", "tds of the current fin. year")

                        # Perform tolerance matching on individual unmatched entries
                        individual_tolerance_matched_df = match_individual_entries_with_tolerance(
                            remaining_unmatched_tds, 
                            remaining_unmatched_zoho, 
                            'Name of Deductor', 
                            'name of the deductor', 
                            'TDS Deposited(Rs.)', 
                            'tds of the current fin. year',
                            tolerance=10
                        )
                        display_dataframe_with_stats(individual_tolerance_matched_df, "Matched Individual Unmatched Entries with Tolerance in TDS and Zoho", "TDS Deposited(Rs.)")

                        # Get remaining unmatched individual entries after removing tolerance matched ones
                        remaining_unmatched_tds_after_tolerance, remaining_unmatched_zoho_after_tolerance = get_remaining_unmatched_entries_with_tolerance(
                            remaining_unmatched_tds, 
                            remaining_unmatched_zoho, 
                            individual_tolerance_matched_df, 
                            'Name of Deductor', 
                            'name of the deductor', 
                            'TDS Deposited(Rs.)', 
                            'tds of the current fin. year'
                        )

                        # Display remaining unmatched individual entries after tolerance matching
                        display_dataframe_with_stats(remaining_unmatched_tds_after_tolerance, "Remaining Unmatched Individual Entries After Tolerance in TDS", "TDS Deposited(Rs.)")
                        display_dataframe_with_stats(remaining_unmatched_zoho_after_tolerance, "Remaining Unmatched Individual Entries After Tolerance in Zoho", "tds of the current fin. year")

                        # Fourth round of matching based on the first three words in the deductor name
                        three_words_matched_df = match_individual_entries_based_on_three_words(
                            remaining_unmatched_tds_after_tolerance, 
                            remaining_unmatched_zoho_after_tolerance, 
                            'Name of Deductor', 
                            'name of the deductor', 
                            'TDS Deposited(Rs.)', 
                            'tds of the current fin. year'
                        )
                        display_dataframe_with_stats(three_words_matched_df, "Matched Individual Unmatched Entries Based on Three Words in Deductor Name", "TDS Deposited(Rs.)")

                        # Get remaining unmatched individual entries after removing matches based on three words in the deductor name
                        remaining_unmatched_tds_after_three_words, remaining_unmatched_zoho_after_three_words = get_remaining_unmatched_entries_after_three_words(
                            remaining_unmatched_tds_after_tolerance, 
                            remaining_unmatched_zoho_after_tolerance, 
                            three_words_matched_df, 
                            'Name of Deductor', 
                            'name of the deductor', 
                            'TDS Deposited(Rs.)', 
                            'tds of the current fin. year'
                        )

                        # Display remaining unmatched individual entries after matching based on three words
                        display_dataframe_with_stats(remaining_unmatched_tds_after_three_words, "Remaining Unmatched Individual Entries After Three Words Matching in TDS", "TDS Deposited(Rs.)")
                        display_dataframe_with_stats(remaining_unmatched_zoho_after_three_words, "Remaining Unmatched Individual Entries After Three Words Matching in Zoho", "tds of the current fin. year")

                        # Perform tolerance matching on individual unmatched entries based on three words in the deductor name
                        three_words_tolerance_matched_df = match_individual_entries_with_tolerance_based_on_three_words(
                            remaining_unmatched_tds_after_three_words, 
                            remaining_unmatched_zoho_after_three_words, 
                            'Name of Deductor', 
                            'name of the deductor', 
                            'TDS Deposited(Rs.)', 
                            'tds of the current fin. year',
                            tolerance=10
                        )
                        display_dataframe_with_stats(three_words_tolerance_matched_df, "Matched Individual Unmatched Entries with Tolerance Based on Three Words in Deductor Name", "TDS Deposited(Rs.)")

                        # Get remaining unmatched individual entries after removing tolerance matched ones based on three words
                        remaining_unmatched_tds_after_tolerance_three_words, remaining_unmatched_zoho_after_tolerance_three_words = get_remaining_unmatched_entries_after_tolerance_three_words(
                            remaining_unmatched_tds_after_three_words, 
                            remaining_unmatched_zoho_after_three_words, 
                            three_words_tolerance_matched_df, 
                            'Name of Deductor', 
                            'name of the deductor', 
                            'TDS Deposited(Rs.)', 
                            'tds of the current fin. year'
                        )

                        # Display remaining unmatched individual entries after tolerance matching based on three words
                        display_dataframe_with_stats(remaining_unmatched_tds_after_tolerance_three_words, "Remaining Unmatched Individual Entries After Tolerance Based on Three Words in TDS", "TDS Deposited(Rs.)")
                        display_dataframe_with_stats(remaining_unmatched_zoho_after_tolerance_three_words, "Remaining Unmatched Individual Entries After Tolerance Based on Three Words in Zoho", "tds of the current fin. year")

                        # Create Final DataFrame with All Matched Entries
                        final_matched_df = pd.concat([
                            exact_matching_df,
                            tolerance_matching_df,
                            individual_matched_df,
                            individual_tolerance_matched_df,
                            three_words_matched_df,
                            three_words_tolerance_matched_df
                        ]).reset_index(drop=True)

                        # Display Final Consolidated DataFrame
                        display_dataframe_with_stats(final_matched_df, "Final Consolidated Matched Entries", "TDS Deposited(Rs.)")

                else:
                    st.warning('Please upload the Zoho Books Excel file to view Zoho data.')

            except Exception as e:
                st.error(f"An error occurred while processing files: {str(e)}")
        else:
            st.sidebar.write("Awaiting file upload...")

if __name__ == "__main__":
    main()
