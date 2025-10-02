"""
Sales and Support Data Analysis for Q3 2024
Analyzes sales transactions and support tickets to derive business insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data():
    """Load sales and support ticket data from CSV files."""
    try:
        sales_df = pd.read_csv('sales_2024Q3.csv')
        support_df = pd.read_csv('support_tickets_2024Q3.csv')
        
        # Convert date columns
        sales_df['order_date'] = pd.to_datetime(sales_df['order_date'])
        support_df['created_at'] = pd.to_datetime(support_df['created_at'])
        
        print(f"✓ Loaded {len(sales_df)} sales records")
        print(f"✓ Loaded {len(support_df)} support tickets\n")
        
        return sales_df, support_df
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return None, None


def analyze_sales_metrics(sales_df):
    """Calculate key sales metrics and performance indicators."""
    print("=== SALES ANALYSIS ===\n")
    
    # Overall metrics
    total_revenue = sales_df['revenue'].sum()
    total_orders = len(sales_df)
    avg_order_value = sales_df['revenue'].mean()
    return_rate = sales_df['is_returned'].mean() * 100
    
    print(f"Total Revenue: ₹{total_revenue:,.2f}")
    print(f"Total Orders: {total_orders}")
    print(f"Average Order Value: ₹{avg_order_value:.2f}")
    print(f"Return Rate: {return_rate:.1f}%\n")
    
    # Revenue by category
    print("Revenue by Category:")
    category_revenue = sales_df.groupby('category')['revenue'].agg(['sum', 'count'])
    category_revenue['avg_order'] = category_revenue['sum'] / category_revenue['count']
    category_revenue = category_revenue.sort_values('sum', ascending=False)
    print(category_revenue)
    print()
    
    # Top products
    print("\nTop 5 Products by Revenue:")
    product_revenue = sales_df.groupby('product')['revenue'].sum().sort_values(ascending=False).head()
    for product, revenue in product_revenue.items():
        print(f"  {product}: ₹{revenue:,.2f}")
    
    # Channel performance
    print("\nChannel Performance:")
    channel_stats = sales_df.groupby('channel').agg({
        'revenue': 'sum',
        'order_id': 'count',
        'is_returned': 'mean'
    }).round(2)
    channel_stats.columns = ['Total Revenue', 'Order Count', 'Return Rate']
    channel_stats['Return Rate'] = (channel_stats['Return Rate'] * 100).round(1)
    print(channel_stats)
    
    # Country breakdown
    print("\nCountry Performance:")
    country_revenue = sales_df.groupby('country')['revenue'].sum().sort_values(ascending=False)
    for country, revenue in country_revenue.items():
        pct = (revenue / total_revenue) * 100
        print(f"  {country}: ₹{revenue:,.2f} ({pct:.1f}%)")
    
    return sales_df


def analyze_support_metrics(support_df):
    """Analyze support ticket patterns and agent performance."""
    print("\n\n=== SUPPORT ANALYSIS ===\n")
    
    # Overall metrics
    total_tickets = len(support_df)
    resolution_rate = support_df['resolved'].mean() * 100
    avg_handle_time = support_df['handle_time_min'].mean()
    
    print(f"Total Tickets: {total_tickets}")
    print(f"Resolution Rate: {resolution_rate:.1f}%")
    print(f"Average Handle Time: {avg_handle_time:.1f} minutes\n")
    
    # Topic distribution
    print("Ticket Distribution by Topic:")
    topic_counts = support_df['topic'].value_counts()
    for topic, count in topic_counts.items():
        pct = (count / total_tickets) * 100
        print(f"  {topic}: {count} ({pct:.1f}%)")
    
    # Priority breakdown
    print("\nPriority Distribution:")
    priority_stats = support_df.groupby('priority').agg({
        'ticket_id': 'count',
        'resolved': 'mean',
        'handle_time_min': 'mean'
    }).round(2)
    priority_stats.columns = ['Count', 'Resolution Rate', 'Avg Handle Time']
    priority_stats['Resolution Rate'] = (priority_stats['Resolution Rate'] * 100).round(1)
    print(priority_stats)
    
    # Agent performance
    print("\nAgent Performance:")
    agent_stats = support_df.groupby('agent').agg({
        'ticket_id': 'count',
        'resolved': lambda x: (x.sum() / len(x) * 100),
        'handle_time_min': 'mean'
    }).round(1)
    agent_stats.columns = ['Tickets', 'Resolution %', 'Avg Time (min)']
    agent_stats = agent_stats.sort_values('Tickets', ascending=False)
    print(agent_stats)
    
    return support_df


def create_visualizations(sales_df, support_df):
    """Generate key visualizations for the analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Revenue trend over time
    ax1 = axes[0, 0]
    daily_revenue = sales_df.groupby('order_date')['revenue'].sum().rolling(7).mean()
    daily_revenue.plot(ax=ax1, linewidth=2.5, color='#2E86AB')
    ax1.set_title('Daily Revenue Trend (7-day MA)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Revenue (₹)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Revenue by channel and category
    ax2 = axes[0, 1]
    channel_category = sales_df.groupby(['channel', 'category'])['revenue'].sum().unstack()
    channel_category.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Revenue by Channel and Category', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Revenue (₹)')
    ax2.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Support ticket volume by topic and priority
    ax3 = axes[1, 0]
    topic_priority = support_df.groupby(['topic', 'priority'])['ticket_id'].count().unstack()
    topic_priority.plot(kind='barh', ax=ax3, width=0.8, stacked=True)
    ax3.set_title('Support Tickets by Topic and Priority', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Number of Tickets')
    ax3.set_ylabel('Topic')
    ax3.legend(title='Priority', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Handle time distribution by topic
    ax4 = axes[1, 1]
    support_df.boxplot(column='handle_time_min', by='topic', ax=ax4)
    ax4.set_title('Handle Time Distribution by Topic', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Topic')
    ax4.set_ylabel('Handle Time (minutes)')
    ax4.tick_params(axis='x', rotation=45)
    plt.suptitle('')  # Remove default title from boxplot
    
    plt.tight_layout()
    plt.savefig('analysis_plots.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualizations saved to 'analysis_plots.png'")
    plt.show()


def generate_insights(sales_df, support_df):
    """Generate business insights and save to file."""
    insights = []
    
    # Calculate key metrics for insights
    total_revenue = sales_df['revenue'].sum()
    services_revenue = sales_df[sales_df['category'] == 'Services']['revenue'].sum()
    services_pct = (services_revenue / total_revenue) * 100
    
    web_revenue = sales_df[sales_df['channel'] == 'Web']['revenue'].sum()
    web_return_rate = sales_df[sales_df['channel'] == 'Web']['is_returned'].mean() * 100
    
    refund_tickets = support_df[support_df['topic'] == 'Refund']
    refund_resolution = refund_tickets['resolved'].mean() * 100
    
    high_priority_resolution = support_df[support_df['priority'] == 'High']['resolved'].mean() * 100
    
    # Write insights
    insights_text = f"""# Q3 2024 Business Insights

## Revenue Performance
Services dominate revenue at {services_pct:.1f}% of total sales (₹{services_revenue:,.0f}), highlighting strong subscription business. 
The Web channel leads with ₹{web_revenue:,.0f} in revenue but shows concerning {web_return_rate:.1f}% return rate, 
suggesting potential UX improvements needed.

## Support Operations
Support handles {len(support_df)} tickets with {support_df['resolved'].mean()*100:.1f}% resolution rate. 
Refund requests show only {refund_resolution:.1f}% resolution, indicating policy or process gaps. 
High-priority tickets achieve {high_priority_resolution:.1f}% resolution, demonstrating effective escalation handling.

## Key Recommendations
1. Investigate Web channel's high return rate through UX analysis
2. Review refund policies to improve {100-refund_resolution:.1f}% unresolved refund tickets
3. Leverage Services category success to expand subscription offerings
4. Optimize support workflows for Pricing/Billing topics (highest volume)
"""
    
    with open('INSIGHTS.md', 'w', encoding='utf-8') as f:
        f.write(insights_text)
    
    print("\n✓ Insights saved to 'INSIGHTS.md'")
    return insights_text


def main():
    """Execute the complete analysis pipeline."""
    print("Starting Q3 2024 Data Analysis...\n")
    
    # Load data
    sales_df, support_df = load_data()
    if sales_df is None or support_df is None:
        return
    
    # Run analyses
    sales_df = analyze_sales_metrics(sales_df)
    support_df = analyze_support_metrics(support_df)
    
    # Generate insights
    generate_insights(sales_df, support_df)
    # Create visualizations
    create_visualizations(sales_df, support_df)
    
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()