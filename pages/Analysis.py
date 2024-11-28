import dash
from dash import html, dcc
from components.Button import Button, ButtonVariants, ButtonSizes
from components.Typography import P, TypographyVariants

dash.register_page(__name__, path="/analysis")

def layout():
    return html.Div([
        html.Div([
            P("Analytic Page", variant=TypographyVariants.HEADING1, className="mb-4"),
            Button("File name.csv", variant=ButtonVariants.PRIMARY, asLink=True, className="mb-4"),

            #Descriptive Analysis
            P("Descriptive Analytics(Using Columns)", variant=TypographyVariants.BODY1, className="mb-2"),
            html.Div([
                P("Mean value", variant=TypographyVariants.BODY2, className="mr-2"),
                Button(children="x.xx", variant=ButtonVariants.PRIMARYGHOST, size=ButtonSizes.SMALL, className="mr-2")
            ], className="flex items-center mb-2"),
            html.Div([
                P("Median value", variant=TypographyVariants.BODY2, className="mr-2"),
                Button(children="x.xx", variant=ButtonVariants.PRIMARYGHOST, size=ButtonSizes.SMALL, className="mr-2")
            ], className="flex items-center mb-2"),
            html.Div([
                P("Mode value", variant=TypographyVariants.BODY2, className="mr-2"),
                Button(children="x", variant=ButtonVariants.PRIMARYGHOST, size=ButtonSizes.SMALL, className="mr-2")
            ], className="flex items-center mb-2"),
            html.Div([
                P("Row counts", variant=TypographyVariants.BODY2, className="mr-2"),
                Button(children="x.xx", variant=ButtonVariants.PRIMARYGHOST, size=ButtonSizes.SMALL, className="mr-2")
            ], className="flex items-center mb-2"),
            html.Div([
                P("Column counts", variant=TypographyVariants.BODY2, className="mr-2"),
                Button(children="x.xx", variant=ButtonVariants.PRIMARYGHOST, size=ButtonSizes.SMALL, className="mr-2")
            ], className="flex items-center mb-2"),

            # Report Using AdaboostClassifier
            html.Div([
                P("Report Using AdaboostClassifier", variant=TypographyVariants.BODY1, className="mb-2"),
                html.Table([
                    html.Tr([
                        html.Th("No", className="p-2 border"),
                        html.Th("Precision", className="p-2 border"),
                        html.Th("Recall", className="p-2 border"),
                        html.Th("F1-Score", className="p-2 border"),
                        html.Th("Support", className="p-2 border")
                    ]),
                    # Sample rows for precision, recall, etc.
                    html.Tr([
                        html.Td("1", className="p-2 border"),
                        html.Td("0.89", className="p-2 border"),
                        html.Td("0.90", className="p-2 border"),
                        html.Td("0.89", className="p-2 border"),
                        html.Td("100", className="p-2 border")
                    ]),
                    html.Tr([
                        html.Td("2", className="p-2 border"),
                        html.Td("0.85", className="p-2 border"),
                        html.Td("0.80", className="p-2 border"),
                        html.Td("0.82", className="p-2 border"),
                        html.Td("120", className="p-2 border")
                    ])
                ], className="w-full border-collapse")
            ], className="mb-4"),

            html.Div(id="settings-container", className="mt-4"),
            html.Div(id="classifier-container", className="mt-4")
        ], className="p-4 bg-white rounded-lg shadow-lg"),

        #Clustering
        html.Div([
            P("K-Means Clustering", variant=TypographyVariants.HEADING2, className="mb-4"),
            html.Div([
                P("Unlabeled Data", variant=TypographyVariants.BODY1, className="mb-2"),
                dcc.Graph(id="unlabeled-data")
            ], className="mb-4"),
            html.Div([
                P("Clustered Data", variant=TypographyVariants.BODY1, className="mb-2"),
                dcc.Graph(id="clustered-data")
            ], className="mb-4")
        ], className="p-4 bg-white rounded-lg shadow-lg")
    ], className="grid grid-cols-1 md:grid-cols-2 gap-4")

