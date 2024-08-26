from IPython.display import HTML, display


def pretty_print_columns_notebook(df: pd.DataFrame):
    # Print columns with types
    div = "<div>"
    div += "<span style='font-size: 16px'>"
    div += f"<pre>{'Column': <40}{'Type'}</pre>"
    div += "<pre>"
    for col in df.columns:
        # Align the columns
        div += f"{str(col): <40}{str(df[col].dtype)} <br>"
    div += "</pre></span></div>"
    display(HTML(div))
