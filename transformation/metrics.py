from constants.constants import INPUT_PATH
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import ast

def load_csv(INPUT_PATH):
    current_directory = os.getcwd()
    filepath = os.path.join(current_directory, INPUT_PATH)
    df = pd.read_csv(f'{filepath}/input.csv')
    return df

def transform_df(df):
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df

def visualize_non_membership_calls(df):
    monthly_counts = df.groupby(['month', 'coaching_call_status']).size().reset_index(name='count')
    monthly_counts['pct_change'] = monthly_counts.groupby('coaching_call_status')['count'].pct_change() * 100
    monthly_counts['pct_change'] = monthly_counts['pct_change'].round().astype('Int64')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly_counts, x='month', y='count', hue='coaching_call_status', marker='o')
    plt.title('Monthly Call Types', fontsize=16)
    plt.xlabel('')
    plt.ylabel('Number of Calls', fontsize=12)
    month_mapping = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sep',
        10: 'Oct'
    }
    unique_months = monthly_counts['month'].unique()
    plt.xticks(ticks=unique_months, labels=[month_mapping[int(month)] for month in unique_months])

    # legend = plt.legend(title='', bbox_to_anchor=(1.05, 1))
    legend = plt.legend(title='', loc='upper left', bbox_to_anchor=(0, 1))

    for _, row in monthly_counts.iterrows():
        # Prepare the count label text
        count_label = f"{row['count']}"
        # Set default color for count
        count_color = 'black'  

        # Prepare the percentage change label text
        if pd.isna(row['pct_change']):
            pct_label = ''
            pct_color = count_color  # Keep color same for NaN
        else:
            pct_label = f"   ({row['pct_change']}%)"
            # Set color based on the percentage change
            pct_color = 'green' if row['pct_change'] > 0 else 'red'

        # Place the count text
        plt.text(row['month'], row['count'], count_label, 
                 ha='center', va='bottom', fontsize=9, color=count_color)
        # Place the percentage change text next to the count
        plt.text(row['month'], row['count'], pct_label, 
                 ha='left', va='bottom', fontsize=9, color=pct_color)
        
    plt.grid(True, which='both', linestyle='--', linewidth=0.2)
    plt.savefig('charts/monthly_coaching_calls.jpg', format='jpg', bbox_inches='tight')
    print('Exported Non Membership Calls Visualization')

def drill_down_non_membership_calls(df):
    filtered_df = df[df['coaching_call_status']=='Non Membership Coaching Call']
    filtered_df['city'] = filtered_df['timezone'].str.split('/').str[1]
    monthly_city_counts = filtered_df.groupby(['month', 'city']).size().reset_index(name='count')
    top_cities_per_month = monthly_city_counts.sort_values(['month', 'count'], ascending=[True, False]) \
        .groupby('month').head(2)
    top_cities_per_month['pct_change'] = top_cities_per_month.groupby('city')['count'].pct_change() * 100
    top_cities_per_month['pct_change'] = top_cities_per_month['pct_change'].round().astype('Int64')    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=top_cities_per_month, x='month', y='count', hue='city', marker='o')
    
    # Set plot title and labels
    plt.title('Top 2 Cities for Non Membership Coaching Calls by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Calls')
    plt.legend(title='City')
    month_mapping = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sep',
        10: 'Oct'
    }
    unique_months = top_cities_per_month['month'].unique()
    plt.xticks(ticks=unique_months, labels=[month_mapping[int(month)] for month in unique_months])
    for _, row in top_cities_per_month.iterrows():
        # Prepare the count label text
        count_label = f"{row['count']}"
        # Set default color for count
        count_color = 'black'  

        # Prepare the percentage change label text
        if pd.isna(row['pct_change']):
            pct_label = ''
            pct_color = count_color  # Keep color same for NaN
        else:
            pct_label = f"   ({row['pct_change']}%)"
            # Set color based on the percentage change
            pct_color = 'green' if row['pct_change'] > 0 else 'red'

        # Place the count text
        plt.text(row['month'], row['count'], count_label, 
                 ha='center', va='bottom', fontsize=9, color=count_color)
        # Place the percentage change text next to the count
        plt.text(row['month'], row['count'], pct_label, 
                 ha='left', va='bottom', fontsize=9, color=pct_color)
    # Show the plot
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.2)
    plt.savefig('charts/monthly_non_membership_calls_per_city.jpg', format='jpg', bbox_inches='tight')
    print('Exported Non Membership Calls Visualization')

def visualize_total_calls_booked(df):
    monthly_counts = df.groupby(['month']).size().reset_index(name='count')
    monthly_counts['pct_change'] = monthly_counts['count'].pct_change() * 100
    monthly_counts['pct_change'] = monthly_counts['pct_change'].round().astype('Int64')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly_counts, x='month', y='count', marker='o')
    plt.title('Monthly Calls Booked', fontsize=16)
    plt.xlabel('')
    plt.ylabel('Number of Calls', fontsize=12)
    month_mapping = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sep',
        10: 'Oct'
    }
    unique_months = monthly_counts['month'].unique()
    plt.xticks(ticks=unique_months, labels=[month_mapping[int(month)] for month in unique_months])

    for _, row in monthly_counts.iterrows():
        # Prepare the count label text
        count_label = f"{row['count']}"
        # Set default color for count
        count_color = 'black'  

        # Prepare the percentage change label text
        if pd.isna(row['pct_change']):
            pct_label = ''
            pct_color = count_color  # Keep color same for NaN
        else:
            pct_label = f"   ({row['pct_change']}%)"
            # Set color based on the percentage change
            pct_color = 'green' if row['pct_change'] > 0 else 'red'

        # Place the count text
        plt.text(row['month'], row['count'], count_label, 
                 ha='center', va='bottom', fontsize=9, color=count_color)
        # Place the percentage change text next to the count
        plt.text(row['month'], row['count'], pct_label, 
                 ha='left', va='bottom', fontsize=9, color=pct_color)
        
    plt.grid(True, which='both', linestyle='--', linewidth=0.2)
    plt.savefig('charts/monthly_calls_booked.jpg', format='jpg', bbox_inches='tight')
    print('Exported Calls Booked Visualization')

def get_coaching_calls_label_name(df):
    def extract_label_info(x):
        try:
            x = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return pd.Series({'label_id': None, 'label_name': None, 'label_color': None})

        if isinstance(x, list) and len(x) > 0:
            if isinstance(x[0], dict):
                return pd.Series({
                    'label_id': x[0].get('id'),
                    'label_name': x[0].get('name'),
                    'label_color': x[0].get('color')
                })
        return pd.Series({'label_id': None, 'label_name': None, 'label_color': None})


    df[['label_id', 'label_name', 'label_color']] = df['labels'].apply(extract_label_info)
    monthly_label_counts = df[df['coaching_call_status']=='Non Membership Coaching Call'].groupby(['month', 'label_name']).size().reset_index(name='count')

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_label_counts, x='month', y='count', hue='label_name', marker='o')

    month_mapping = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sep',
        10: 'Oct'
    }
    unique_months = monthly_label_counts['month'].unique()
    plt.xticks(ticks=unique_months, labels=[month_mapping[int(month)] for month in unique_months])
    
    for _, row in monthly_label_counts.iterrows():
        # Prepare the count label text
        count_label = f"{row['count']}"
        # Set default color for count
        count_color = 'black'  

        # Place the count text
        plt.text(row['month'], row['count'], count_label, 
                 ha='center', va='bottom', fontsize=9, color=count_color)
        
    plt.title('Non Membership Call Sub Types per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Calls')
    plt.legend(title='Call Sub Type')

    # Show the plot
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.2)
    plt.savefig('charts/monthly_call_sub_type.jpg', format='jpg', bbox_inches='tight')
    print('Exported Call Sub Type Visualization')

def get_non_membership_call_breakdown(df):
    filtered_df = df[df['coaching_call_status']=='Non Membership Coaching Call']
    monthly_counts = filtered_df.groupby(['month', 'type']).size().reset_index(name='count')
    monthly_counts['pct_change'] = monthly_counts.groupby('type')['count'].pct_change() * 100
    monthly_counts['pct_change'] = monthly_counts['pct_change'].round().astype('Int64')
    top_5_counts = monthly_counts.nlargest(10, 'count')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=top_5_counts, x='month', y='count', hue='type', marker='o')
    plt.title('Monthly Call Types', fontsize=16)
    plt.xlabel('')
    plt.ylabel('Number of Calls', fontsize=12)
    month_mapping = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sep',
        10: 'Oct'
    }
    unique_months = top_5_counts['month'].unique()
    plt.xticks(ticks=unique_months, labels=[month_mapping[int(month)] for month in unique_months])

    # legend = plt.legend(title='', bbox_to_anchor=(1.05, 1))
    legend = plt.legend(title='', loc='upper left', bbox_to_anchor=(0, 1))

    for _, row in top_5_counts.iterrows():
        # Prepare the count label text
        count_label = f"{row['count']}"
        # Set default color for count
        count_color = 'black'  

        # Place the count text
        plt.text(row['month'], row['count'], count_label, 
                 ha='center', va='bottom', fontsize=9, color=count_color)
        
    plt.grid(True, which='both', linestyle='--', linewidth=0.2)
    plt.savefig('charts/monthly_non_membership_call_types.jpg', format='jpg', bbox_inches='tight')
    print('Exported Non Membership Call Type Visualization')

    return
def get_customer_activity_log(df):
    df['coaching_call_status'] = np.where(df['type'] == 'Membership Coaching Call', 
                                       'Membership Coaching Call', 
                                       'Non Membership Coaching Call')
    
    visualize_non_membership_calls(df)
    visualize_total_calls_booked(df)
    drill_down_non_membership_calls(df)
    get_coaching_calls_label_name(df)
    get_non_membership_call_breakdown(df)

if __name__=='__main__':
    df = load_csv(INPUT_PATH)
    filtered_df = transform_df(df)
    get_customer_activity_log(filtered_df)