from dash import html, callback, Input, Output, State, dcc
from components.Typography import P
from components.Button import Button

import dash_daq as daq
import pandas as pd


def DataDialog():
    return html.Div(
        id="data-dialog",
        children=[
            dcc.Store(id="used-col-row", storage_type="local"),
            dcc.Store(id="file-store", storage_type="local"),
            P("Analysis Setting", variant="body1"),
            html.P(id="form-output"),
            html.Button(
                id="close_btn",
                children=[html.Img(src="assets/images/cross.svg", className="size-6")],
                className="absolute right-3 top-3",
            ),
            daq.ToggleSwitch(
                id="use-row",
                value=False,
                label="Using Row",
                size=40,
                labelPosition="bottom",
            ),
            dcc.Dropdown(id="col-names", className="my-2"),
            dcc.Checklist(id="checklist"),
            html.P(id="checked"),
        ],
        className="w-[814px] h-[400px] overflow-auto hidden flex flex-col gap-3 fixed left-[50%] top-[50%] -translate-x-[50%] -translate-y-[50%] p-5 rounded-xl bg-[#D2E9E9] z-[120] duration-300",
        style={"boxShadow": "0 0 30px 0px rgba(0, 0, 0, 0.50)", "display": "none"},
    )


@callback(
    Output("data-dialog", "style", allow_duplicate=True),
    Input("close_btn", "n_clicks"),
    prevent_initial_call=True,
)
def closeDialog(n_clicks):
    return {"display": "none"}


@callback(
    Output("col-names", "options"),
    Input("file-store", "data"),
    prevent_initial_call=True,
)
def getColNames(file):
    if file and file["content"]:
        df = pd.DataFrame(file["content"])
        cols = []
        for col in df.columns:
            cols.append(col)
        return cols
    else:
        return []


@callback(
    Output("checklist", "options"),
    Output("col-names", "style"),
    Output("use-row", "label"),
    Input("file-store", "data"),
    Input("use-row", "value"),
    Input("col-names", "value"),
    prevent_initial_call=True,
)
def process_form(file, useRow, colName):
    checklists = []
    if file and file["content"]:
        df = pd.DataFrame(file["content"])
        if useRow is True:
            for col in df.columns:
                checklists.append(col)
        else:
            new_df = df
            if colName != None:
                new_df.set_index(colName)
            for row in new_df.set_index("animal_name").axes[0].tolist():
                checklists.append(row)
        return (
            checklists,
            {"display": "none"} if useRow else {"display": "block"},
            "Using Rows" if useRow else "Using Columns",
        )
    else:
        return checklists, {}, "Using Rows"


@callback(
    Output("used-col-row", "data"),
    Input("checklist", "value"),
    Input("use-row", "value"),
    Input("file-store", "data"),
)
def updateColRow(checkedValues, useRow, data):
    if useRow:
        return {"useRow": True, "values": checkedValues}
    else:
        return {"useRow": False, "values": checkedValues}
