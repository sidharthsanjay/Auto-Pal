# Auto Pal Documentation 
Auto Pal is a multi-functional Streamlit-based dashboard application built for property data validation and 
visualization. It integrates with MongoDB and provides tools for: 
* Manual data validation and logging 
* Convertible status analysis 
* Market undervalue insights 
* Visual dashboards for updates and statuses 
  
## Project Structure 
```
  AutoPal/ 
  │ 
  ├── Home.py               - Main app script with routing and multi-page logic       
  ├── engine.py             - Standalone dashboard logic for status-based pie charts     
  ├── logo.png              - Branding logo shown in sidebar
  └── updation_data.csv     - Log of updates and validations (dynamically generated)
```

## How to Run 
  Prerequisites: 
   * Python 3.8+ 
  To run the app: 
   * streamlit run Home.py
    
## MongoDB Setup 
  * Connection URI is hardcoded in both Home.py and engine.py 
  * MongoDB Connection used: 
    * sales_data: Main property data 
    * update_logs: Validation logs 
    * dashboard_collection: For status dashboard summaries 

## Page Descriptions 
  1. Validation Tool(Home.py) 
*Filters data by: 
      * AI Type (Potential, Under Value) 
      * Date range 
      * Source & Property Type 
    * Allows manual navigation and validation for each property 
    * Logs changes to MongoDB and appends to updation_data.csv 
  2. Validation Convertible Status 
    * Shows a pie chart of convertible status values that are not yet updated 
    * Pulls live data from MongoDB 
  3. Validation Dashboard 
    * Automatically loads filters passed from the Validation Tool 
    * Visualizations: 
      * Property type histogram 
      * Bedroom box plot 
      * Listing trends (line chart) 
      * Bedroom vs postcode heatmap 
  4. Engine Dashboard(engine.py) 
    * Shows status breakdown for the selected data source 
    * Displays: 
      * Pie chart of status (Completed/Failed/Null) 
      * Pie chart of update dates 
      * Filterable table view
