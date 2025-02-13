import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium", app_title="Marimo Viewer: Cirro")


@app.cell
def _(mo):
    mo.md(r"""# Pangenome Viewer (gig-map)""")
    return


@app.cell
def _():
    # Load the marimo library in a dedicated cell for efficiency
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # If the script is running in WASM (instead of local development mode), load micropip
    import sys
    if "pyodide" in sys.modules:
        import micropip
        running_in_wasm = True
    else:
        micropip = None
        running_in_wasm = False
    return micropip, running_in_wasm, sys


@app.cell
async def _(micropip, mo, running_in_wasm):
    with mo.status.spinner("Loading dependencies"):
        # If we are running in WASM, some dependencies need to be set up appropriately.
        # This is really just aligning the needs of the app with the default library versions
        # that come when a marimo app loads in WASM.
        if running_in_wasm:
            print("Installing via micropip")
            # Downgrade plotly to avoid the use of narwhals
            await micropip.install("plotly<6.0.0")
            await micropip.install("ssl")
            micropip.uninstall("urllib3")
            micropip.uninstall("httpx")
            await micropip.install(["urllib3==2.3.0"])
            await micropip.install([
                "boto3==1.36.3",
                "botocore==1.36.3"
            ], verbose=True)
            await micropip.install(["cirro[pyodide]>=1.2.16"], verbose=True)
            await micropip.install("umap-learn", verbose=True)

        from io import StringIO, BytesIO
        from queue import Queue
        from time import sleep
        from typing import Dict, Optional
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        import numpy as np
        from functools import lru_cache
        import base64
        from urllib.parse import quote_plus

        from cirro import DataPortalLogin, DataPortalDataset
        from cirro.services.file import FileService
        from cirro.sdk.file import DataPortalFile
        from cirro.config import list_tenants
        from scipy import cluster

        # A patch to the Cirro client library is applied when running in WASM
        if running_in_wasm:
            from cirro.helpers import pyodide_patch_all
            pyodide_patch_all()
    return (
        BytesIO,
        DataPortalDataset,
        DataPortalFile,
        DataPortalLogin,
        Dict,
        FileService,
        Optional,
        Queue,
        StringIO,
        base64,
        cluster,
        go,
        list_tenants,
        lru_cache,
        make_subplots,
        np,
        pd,
        px,
        pyodide_patch_all,
        quote_plus,
        sleep,
    )


@app.cell
def _(DataPortalDataset):
    # Define the types of datasets which can be read in
    # This is used to filter the dataset selector, below
    def filter_datasets_pangenome(dataset: DataPortalDataset):
        return "hutch-gig-map-build-pangenome" in dataset.process_id and dataset.status == "COMPLETED"
    return (filter_datasets_pangenome,)


@app.cell
def _(mo):
    # Get and set the query parameters
    query_params = mo.query_params()
    return (query_params,)


@app.cell
def _(list_tenants):
    # Get the tenants (organizations) available in Cirro
    tenants_by_name = {i["displayName"]: i for i in list_tenants()}
    tenants_by_domain = {i["domain"]: i for i in list_tenants()}


    def domain_to_name(domain):
        return tenants_by_domain.get(domain, {}).get("displayName")


    def name_to_domain(name):
        return tenants_by_name.get(name, {}).get("domain")
    return (
        domain_to_name,
        name_to_domain,
        tenants_by_domain,
        tenants_by_name,
    )


@app.cell
def _(mo):
    mo.md(r"""## Load Data""")
    return


@app.cell
def _(mo):
    # Use a state element to manage the Cirro client object
    get_client, set_client = mo.state(None)
    return get_client, set_client


@app.cell
def _(domain_to_name, mo, query_params, tenants_by_name):
    # Let the user select which tenant to log in to (using displayName)
    domain_ui = mo.ui.dropdown(
        options=tenants_by_name,
        value=domain_to_name(query_params.get("domain")),
        on_change=lambda i: query_params.set("domain", i["domain"]),
        label="Load Data from Cirro",
    )
    domain_ui
    return (domain_ui,)


@app.cell
def _(DataPortalLogin, domain_ui, get_client, mo):
    # If the user is not yet logged in, and a domain is selected, then give the user instructions for logging in
    # The configuration of this cell and the two below it serve the function of:
    #   1. Showing the user the login instructions if they have selected a Cirro domain
    #   2. Removing the login instructions as soon as they have completed the login flow
    if get_client() is None and domain_ui.value is not None:
        with mo.status.spinner("Authenticating"):
            # Use device code authorization to log in to Cirro
            cirro_login = DataPortalLogin(base_url=domain_ui.value["domain"])
            cirro_login_ui = mo.md(cirro_login.auth_message_markdown)
    else:
        cirro_login = None
        cirro_login_ui = None

    mo.stop(cirro_login is None)
    cirro_login_ui
    return cirro_login, cirro_login_ui


@app.cell
def _(cirro_login, set_client):
    # Once the user logs in, set the state for the client object
    set_client(cirro_login.await_completion())
    return


@app.cell
def _(get_client, mo):
    # Get the Cirro client object (but only take action if the user selected Cirro as the input)
    client = get_client()
    mo.stop(client is None)
    return (client,)


@app.cell
def _():
    # Helper functions for dealing with lists of objects that may be accessed by id or name
    def id_to_name(obj_list: list, id: str) -> str:
        if obj_list is not None:
            return {i.id: i.name for i in obj_list}.get(id)


    def name_to_id(obj_list: list) -> dict:
        if obj_list is not None:
            return {i.name: i.id for i in obj_list}
        else:
            return {}
    return id_to_name, name_to_id


@app.cell
def _(client):
    # Set the list of projects available to the user
    projects = client.list_projects()
    projects.sort(key=lambda i: i.name)
    return (projects,)


@app.cell
def _(id_to_name, mo, name_to_id, projects, query_params):
    # Let the user select which project to get data from
    project_ui = mo.ui.dropdown(
        value=id_to_name(projects, query_params.get("project")),
        options=name_to_id(projects),
        on_change=lambda i: query_params.set("project", i)
    )
    project_ui
    return (project_ui,)


@app.cell
def _(client, filter_datasets_pangenome, mo, project_ui):
    # Stop if the user has not selected a project
    mo.stop(project_ui.value is None)

    # Get the list of pangenome datasets available to the user
    pangenome_datasets = [
        dataset
        for dataset in client.get_project_by_id(project_ui.value).list_datasets()
        if filter_datasets_pangenome(dataset)
    ]
    pangenome_datasets.sort(key=lambda ds: ds.created_at, reverse=True)
    return (pangenome_datasets,)


@app.cell
def _(id_to_name, mo, name_to_id, pangenome_datasets, query_params):
    # Let the user select which pangenome dataset to get data from
    pangenome_dataset_ui = mo.ui.dropdown(
        label="Select Pangenome:",
        value=id_to_name(pangenome_datasets, query_params.get("dataset")),
        options=name_to_id(pangenome_datasets),
        on_change=lambda i: query_params.set("dataset", i)
    )
    pangenome_dataset_ui
    return (pangenome_dataset_ui,)


@app.cell
def _(client, mo, pangenome_dataset_ui, project_ui):
    # Stop if the user has not selected a dataset
    mo.stop(pangenome_dataset_ui.value is None)

    # Get the selected dataset
    pangenome_dataset = (
        client
        .get_project_by_id(project_ui.value)
        .get_dataset_by_id(pangenome_dataset_ui.value)
    )
    return (pangenome_dataset,)


@app.cell
def _(DataPortalDataset, Dict, mo, pangenome_dataset, pd):
    # Define an object with all of the information for a pangenome
    class Pangenome:
        ds: DataPortalDataset
        bin_contents: Dict[str, pd.DataFrame]
        genome_contents: pd.DataFrame
        bin_size: pd.Series

        def __init__(self, ds: DataPortalDataset):
            self.ds = ds

            # Read in the table of which genes are in which bins
            self.bin_contents = {
                bin: d.drop(columns=['bin'])
                for bin, d in self.read_csv("data/bin_pangenome/gene_bins.csv").groupby("bin")
            }
            # Number of genes per bin
            self.bin_size = pd.Series({bin: d.shape[0] for bin, d in self.bin_contents.items()}).sort_values(ascending=False)

            # Read in the table listing which genomes contain which bins
            self.genome_contents = self.read_csv(
                "data/bin_pangenome/genome_content.long.csv",
                usecols=["bin", "genome", "n_genes_detected", "prop_genes_detected", "organism_organismName", "assemblyInfo_biosample_strain"]
            )

        def read_csv(self, fp: str, **kwargs):
            return self.ds.list_files().get_by_id(fp).read_csv(**kwargs)

    with mo.status.spinner("Loading data..."):
        pg = Pangenome(pangenome_dataset)

    return Pangenome, pg


@app.cell
def _(mo):
    # The user can apply filters for visualizing the pangenome
    pangenome_args = (
        mo.md("""
    ### Pangenome Options

    - {min_prop}
    - {min_bin_size}
        """)
        .batch(
            min_prop=mo.ui.number(
                label="Minimum Proportion of Genes in Bin:",
                start=0.01,
                stop=1.0,
                value=0.9
            ),
            min_bin_size=mo.ui.number(
                label="Minimum Bin Size (# of Genes):",
                start=1,
                value=5
            )
        )
    )
    pangenome_args
    return (pangenome_args,)


@app.cell
def _(Pangenome, cluster, make_subplots, mo, pangenome_args, pd, pg):
    # Based on the inputs, format and present a display
    class PangenomeDisplay:
        # Index: Genomes
        # Columns: Bins
        # Values: 1/0
        wide: pd.DataFrame

        pg: Pangenome

        def __init__(self, pg: Pangenome, min_prop: float, min_bin_size: int):
            self.pg = pg

            # Make a wide DataFrame with the bins that each genome meets the threshold for
            self.wide = (
                pg.genome_contents
                .query(f"prop_genes_detected >= {min_prop}")
                .assign(present=1)
                .assign(bin_size=lambda d: d["bin"].apply(self.pg.bin_size.get))
                .query(f"bin_size >= {min_bin_size}")
                .pivot(index="genome", columns="bin", values="present")
                .fillna(0)
                .map(int)
            )

        def plot(self):
            """Make a heatmap showing which bins are present in which genomes."""

            genome_order = sort_axis(self.wide)
            bin_order = sort_axis(self.wide.T)

            fig = make_subplots(
                shared_xaxes=True,
                shared_yaxes=True,
                rows=2,
                # cols=2,
                start_cell="bottom-left",
                # column_widths=[2, 1],
                row_heights=[2, 1],
                horizontal_spacing=0.01,
                vertical_spacing=0.01
            )
            fig.add_heatmap(
                x=bin_order,
                y=genome_order,
                z=self.wide.reindex(index=genome_order, columns=bin_order),
                text=[
                    [
                        f"{genome}<br>{bin}<extra></extra>"
                        for bin in bin_order
                    ]
                    for genome in genome_order
                ],
                hovertemplate="%{text}<extra></extra>",
                colorscale="Blues",
                row=1,
                col=1,
                showscale=False,
            )
            fig.add_bar(
                y=self.pg.bin_size.reindex(bin_order),
                row=2,
                col=1,
                showlegend=False,
                text=[
                    f"{bin_id}<br>{self.pg.bin_size[bin_id]:,} Genes"
                    for bin_id in bin_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis2=dict(title_text="Bin Size<br>(# of Genes)", type="log"),
            )
            return fig


    def sort_axis(df: pd.DataFrame):
        return df.index.values[
            cluster.hierarchy.leaves_list(
                cluster.hierarchy.linkage(
                    df.values,
                    method="average",
                    metric="jaccard"
                )
            )
        ]

    with mo.status.spinner("Computing layout..."):
        pg_display = PangenomeDisplay(pg, **pangenome_args.value)

    return PangenomeDisplay, pg_display, sort_axis


@app.cell
def _(pg_display):
    pg_display.plot()
    return


@app.cell
def _(mo, pg):
    # Show the user information about a specific bin
    display_bin_args = (
        mo.md(
            """
    ### Show Bin Information

    {bin_id}
            """
        )
        .batch(
            bin_id=mo.ui.dropdown(
                label="Display:",
                options=pg.bin_size.index.values,
                value=pg.bin_size.index.values[0]
            )
        )
    )
    display_bin_args
    return (display_bin_args,)


@app.cell
def _(mo, pg):
    def display_bin(bin_id):
        # Show the list of genes in the bin
        return mo.vstack([
            (
                pg.bin_contents[bin_id]
                .set_index("gene_id")
                .rename(
                    columns=dict(
                        combined_name="Gene Annotation",
                        n_genomes="Number of Genomes"
                    )
                )
            ),
            (
                pg.genome_contents
                .query(f"bin == '{bin_id}'")
                .sort_values(by="n_genes_detected", ascending=False)
                .rename(columns=dict(
                    organism_organismName="Organism Name",
                    genome="Genome ID",
                    prop_genes_detected="Proportion Detected",
                    n_genes_detected="Number of Genes Detected",
                    assemblyInfo_biosample_strain="Strain"
                ))
                .set_index("Genome ID")
                .drop(columns=["bin"])
            )
        ])
    return (display_bin,)


@app.cell
def _(display_bin, display_bin_args):
    display_bin(**display_bin_args.value)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
