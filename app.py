from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import dash_table
import getDataFromYahoo as abc
# Initialize the Dash app with a Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
# App layout
app.layout = dbc.Container(
    [
        html.H1("Asset Allocation Dashboard", className="mt-4 mb-4"),
        html.P("Enter asset allocations as percentages for the following indices:", className="mb-3"),
        dbc.Form(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Label("S&P 500 (%)"), width=2),
                        dbc.Col(dcc.Input(id="input-sp500", type="number", placeholder="0-100", min=0, max=100), width=4),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("MSCI ACWI (%)"), width=2),
                        dbc.Col(dcc.Input(id="input-acwi", type="number", placeholder="0-100", min=0, max=100), width=4),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("MSCI EM (%)"), width=2),
                        dbc.Col(dcc.Input(id="input-em", type="number", placeholder="0-100", min=0, max=100), width=4),
                    ],
                    className="mb-3",
                ),
                dbc.Button("Submit", id="submit-button", color="primary", className="mt-3"),
            ]
        ),
        html.Div(id="output-summary", className="mt-4"),
        html.Div(id="portfolio-table", className="mt-4"),  # Placeholder for the table
    ],
    fluid=True,
)

# Backend callback for the Submit button
@app.callback(
    [Output("output-summary", "children"), Output("portfolio-table", "children")],
    [Input("submit-button", "n_clicks")],
    [State("input-sp500", "value"), State("input-acwi", "value"), State("input-em", "value")],
)
def process_allocation(n_clicks, sp500, acwi, em):
    if n_clicks is None:
        return "", ""

    # Example logic: Call backend function to get portfolio returns
    portfolio_returns = abc.printPortFolioReturns()  # Replace with actual function

    # Convert DataFrame to dictionary for Dash DataTable
    portfolio_table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in portfolio_returns.columns],
        data=portfolio_returns.reset_index().to_dict("records"),  # Convert DataFrame to list of dicts
        style_table={"overflowX": "auto", "maxHeight": "400px"},
        style_cell={"textAlign": "center", "padding": "5px"},
        style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"},
        page_size=10,  # Paginate to display 10 rows at a time
    )

    # Display the allocation summary and portfolio table
    summary = html.Div(
        [
            html.H5("Allocation Summary:"),
            html.P(f"S&P 500: {sp500}%"),
            html.P(f"MSCI ACWI: {acwi}%"),
            html.P(f"MSCI EM: {em}%"),
        ]
    )
    return summary, portfolio_table



# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
