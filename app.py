import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium", app_title="Pangenome Viewer ")


@app.cell
def _(mo):
    mo.sidebar(
        [
            mo.nav_menu(
                {
                    "#pangenome-viewer-gig-map": f"{mo.icon('lucide:info')} About",
                    "#connect-to-database": f"{mo.icon('lucide:database')} Connect",
                    "#inspect-pangenome": f"{mo.icon('lucide:map')} Inspect Pangenome",
                    "#inspect-gene-bin": f"{mo.icon('lucide:box')} Inspect Gene Bin",
                    "#compare-gene-bins": f"{mo.icon('lucide:boxes')} Compare Gene Bins",
                    "#inspect-metagenomes": f"{mo.icon('lucide:test-tube-diagonal')} Inspect Metagenomes",
                    "#compare-metagenomes": f"{mo.icon('lucide:test-tubes')} Compare Metagenomes",
                    "#settings": f"{mo.icon('lucide:settings')} Settings"
                },
                orientation="vertical",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Pangenome Viewer (gig-map)

    The [gig-map (for "genes-in-genomes map")](https://github.com/FredHutch/gig-map/wiki) tool
    is used to analyze collections of genomes ("pan-genomes") according to which genes are shared
    among different groups of genomes.
    Genes are organized into **bins** -- groups of genes which are all generally found in the same genomes.

    Each bin represents a group of genes which is either (a) preserved together within a lineage, or (b)
    consistently transferred together between lineages.
    Biologically, these tend to be species/genus core genomes, operons, genomic islands, plasmids, bacteriophages, etc.
    """
    )
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
    return micropip, running_in_wasm


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
            await micropip.install("urllib3==2.3.0")
            micropip.uninstall("requests")
            await micropip.install("requests==2.32.3")
            await micropip.install("httpx==0.26.0")
            await micropip.install("botocore==1.37.3")
            await micropip.install("jmespath==1.0.1")
            await micropip.install("s3transfer==0.11.3")
            await micropip.install("boto3==1.37.3")
            await micropip.install("aiobotocore==2.22.0")
            await micropip.install("cirro[pyodide]==1.5.4")  
            await micropip.install("anndata==0.11.3")  

        from typing import Dict, Optional, List, Tuple, Set
        from functools import lru_cache
        from collections import defaultdict
        from itertools import groupby
        from copy import copy

        from cirro import DataPortalLogin, DataPortalDataset
        from cirro.config import list_tenants
        from cirro.sdk.exceptions import DataPortalAssetNotFound

        # A patch to the Cirro client library is applied when running in WASM
        if running_in_wasm:
            from cirro.helpers import pyodide_patch_all
            pyodide_patch_all()

    return (
        DataPortalAssetNotFound,
        DataPortalDataset,
        DataPortalLogin,
        Dict,
        List,
        Optional,
        Set,
        Tuple,
        copy,
        defaultdict,
        groupby,
        list_tenants,
        lru_cache,
    )


@app.cell
def _(DataPortalDataset):
    # Define the types of datasets which can be read in containing pangenome information
    # This is used to filter the dataset selector, below
    def filter_datasets_pangenome(dataset: DataPortalDataset):
        return "hutch-gig-map-build-pangenome" in dataset.process_id and dataset.status == "COMPLETED"
    return (filter_datasets_pangenome,)


@app.cell
def _(DataPortalDataset):
    # Define the types of datasets which can be read in
    # This is used to filter the dataset selector, below
    def filter_datasets_metagenome(dataset: DataPortalDataset):
        return "hutch-gig-map-align-pangenome" in dataset.process_id and dataset.status == "COMPLETED"
    return (filter_datasets_metagenome,)


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
    return domain_to_name, tenants_by_name


@app.cell
def _(mo):
    mo.md(r"""## Connect to Database""")
    return


@app.cell
def _(mo):
    # Use a state element to manage the Cirro client object
    get_client, set_client = mo.state(None)
    return get_client, set_client


@app.cell
def select_tenant(domain_to_name, mo, query_params, tenants_by_name):
    # Let the user select which tenant to log in to (using displayName)
    domain_ui = mo.ui.dropdown(
        options=tenants_by_name,
        value=domain_to_name(query_params.get("domain")),
        label="Load Data from Cirro",
    )
    domain_ui
    return (domain_ui,)


@app.cell
def get_client(DataPortalLogin, domain_ui, get_client, mo):
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
    return (cirro_login,)


@app.cell
def set_client(cirro_login, set_client):
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
def _(get_client, mo):
    mo.stop(get_client() is not None)
    mo.md("*_Log in to view data_*")
    return


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
        label="Select Project:",
        value=id_to_name(projects, query_params.get("project")),
        options=name_to_id(projects),
        searchable=True
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
def _(mo):
    mo.md(r"""## Inspect Pangenome""")
    return


@app.cell
def _(id_to_name, mo, name_to_id, pangenome_datasets, query_params):
    # Let the user select which pangenome dataset to get data from
    pangenome_dataset_ui = mo.ui.dropdown(
        label="Select Pangenome:",
        value=id_to_name(pangenome_datasets, query_params.get("inspect_pangenome")),
        options=name_to_id(pangenome_datasets),
        searchable=True
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
def _(mo, pangenome_dataset):
    mo.accordion({
        "Analysis Settings": mo.md("""
    | Analysis Settings | |
    | --- | --- |
    | Gene Clustering Similarity | {gene_catalog_cluster_similarity}% |
    | Gene Clustering Overlap | {gene_catalog_cluster_similarity}% |
    | Minimum Gene Length | {gene_catalog_min_gene_length:,}aa |
    | Alignment Minimum Coverage | {alignment_min_coverage}% |
    | Alignment Minimum Identity | {alignment_min_identity}% |
    | Gene Binning Threshold | {clustering_max_dist_genes}% |
    | Minimum Gene Bin Size | {clustering_min_bin_size} |
    | Minimum Genomes Per Gene | {clustering_min_genomes_per_gene} |

        """.format(
            **{
                f"{group}_{kw}": (
                    int(val * 100) if kw in ["cluster_similarity", "cluster_coverage", "max_dist_genes"]
                    else val
                )
                for group, group_attrs in pangenome_dataset.params.additional_properties.items()
                for kw, val in group_attrs.items()
            }
        ))
    })
    return


@app.cell
def _(mo):
    with mo.status.spinner("Loading dependencies"):
        import pandas as pd
        import pyarrow
        from scipy import cluster, spatial, stats
        from anndata import AnnData
        import sklearn
        from plotly import graph_objects as go
        from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
        from Bio.Phylo.BaseTree import Tree, Clade
        from Bio import Phylo

    return (
        AnnData,
        Clade,
        DistanceMatrix,
        DistanceTreeConstructor,
        Tree,
        cluster,
        go,
        pd,
        sklearn,
        spatial,
        stats,
    )


@app.cell
def _(client, lru_cache, mo):
    @lru_cache
    def read_csv_cached(
        project_id: str,
        dataset_id: str,
        filepath: str,
        **kwargs
    ):
        ds = client.get_dataset(project_id, dataset_id)
        with mo.status.spinner(f"Reading File: {filepath} ({ds.name})"):
            return ds.list_files().get_by_id(filepath).read_csv(**kwargs)
    return (read_csv_cached,)


@app.cell
def _(client, lru_cache, mo):
    @lru_cache
    def read_text_cached(
        project_id: str,
        dataset_id: str,
        filepath: str,
        **kwargs
    ):
        ds = client.get_dataset(project_id, dataset_id)
        with mo.status.spinner(f"Reading File: {filepath} ({ds.name})"):
            return ds.list_files().get_by_id(filepath).read(**kwargs)

    return (read_text_cached,)


@app.cell
def _(
    Clade,
    Dict,
    DistanceMatrix,
    DistanceTreeConstructor,
    Tree,
    defaultdict,
    go,
    make_subplots,
    mo,
    np,
    pd,
    stats,
):
    class Phylogeny:
        """
        Helper object used to coordinate a phylogeny.
        dm is a distance matrix of SNP rates provided in pandas DataFrame format.
        """
        tree: Tree
        dm: DistanceMatrix
        distances: Dict[str, Dict[str, float]]

        def __init__(
            self,
            name: str,
            snp_rate: pd.DataFrame
        ):
            self.name = name

            # Make a distance matrix in BioPython format
            self.dm = DistanceMatrix(
                names=list(snp_rate.index.values),
                matrix=[
                    l[:(i+1)]
                    for i, l in enumerate(snp_rate.values.tolist())
                ]
            )

            # Store the distances as a dict of dicts
            self.distances = defaultdict(dict)
            for ix1, name1 in enumerate(self.dm.names):
                for ix2, name2 in enumerate(self.dm.names):
                    if ix1 < ix2:
                        self.distances[name1][name2] = self.dm.matrix[ix2][ix1]
                        self.distances[name2][name1] = self.dm.matrix[ix2][ix1]

            # Calculate the neighbor joining tree
            constructor = DistanceTreeConstructor()
            self.tree = constructor.nj(self.dm)
            self.tree.root_at_midpoint()

            # Get the children of each node
            self.children = {}
            self._get_children(self.tree.clade)

            self.find_coords()

        def find_coords(self):

            # Get the X-Y position of each node (framing the whole tree from 0-1)
            self.coords = {}
            self._add_coord(self.tree.clade, 0., 1.)

        def _add_coord(self, clade: Clade, start: float, stop: float):

            # If the clade is terminal, put it in the middle
            if clade.is_terminal():
                y=np.mean([start, stop])

            # If it's an internal node
            else:
                # See how much space we have to work with
                range = stop - start

                # Get the relative sizes of each child clade
                child_sizes = [
                    len(child.get_terminals())
                    for child in clade.clades
                ]

                # Position this clade between the two
                y = start + (range * child_sizes[0] / np.sum(child_sizes))

                self._add_coord(clade.clades[0], start, y)
                self._add_coord(clade.clades[1], y, stop)

            self.coords[clade.name if clade.name is not None else 'root'] = dict(
                x=self.tree.depths().get(clade, 0),
                y=y
            )            

        def _get_children(self, clade):
            self.children['root' if clade.name is None else clade.name] = [child.name for child in clade.clades]
            for child in clade.clades:
                if not child.is_terminal():
                    self._get_children(child)

        def plot(self):
            # Set up a figure
            fig = make_subplots(rows=1, cols=1)

            self._plot_lines(fig)
            self._plot_points(fig, mode="markers+text")
            fig.update_layout(
                template="simple_white",
                yaxis=dict(
                    visible=False,
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                ),
                xaxis=dict(
                    automargin=True,
                    title_text="SNP Rate"
                ),
                margin=dict(l=100, r=400, b=100, t=100),
                title_text=self.name,
            )
            return mo.ui.plotly(fig)

        def _plot_lines(self, fig, row=1, col=1):

            # For each internal node, draw a line to its children
            for parent, children in self.children.items():
                for child in children:
                    fig.add_trace(
                        go.Scatter(
                            x=[self.coords[parent]['x'], self.coords[parent]['x'], self.coords[child]['x']],
                            y=[self.coords[parent]['y'], self.coords[child]['y'], self.coords[child]['y']],
                            mode="lines",
                            showlegend=False,
                            line_color="black"
                        ),
                        row=row,
                        col=col
                    )

        def _plot_points(self, fig, mode: str, row=1, col=1):
            # Draw each terminal node
            fig.add_trace(
                go.Scatter(
                    x=self._get_coord('x'),
                    y=self._get_coord('y'),
                    text=[node.name for node in self.tree.get_terminals()],
                    mode=mode,
                    showlegend=False,
                    textposition="middle right",
                    marker_color="black",
                    cliponaxis=False
                ),
                row=row,
                col=col
            )

        def _get_coord(self, kw: str):
            """Get a particular value for every item in the tree."""
            return [
                self.coords[node.name][kw]
                for node in self.tree.get_terminals()
            ]

        def _plot_tracer(self, fig, use_nodes, row=1, col=1):
            # Draw a line from each terminal node to the edge of the graph
            edge = np.max(self._get_coord("x"))
            for node_name in use_nodes:
                fig.add_trace(
                    go.Scatter(
                        x=[self.coords[node_name]['x'], edge],
                        y=[self.coords[node_name]['y'], self.coords[node_name]['y']],
                        mode="lines",
                        showlegend=False,
                        line=dict(dash='dot', color="gray"),
                        cliponaxis=False
                    ),
                    row=row,
                    col=col
                )

        def align_trees(self, comp: 'Phylogeny'):

            made_switch = True
            for _ in range(3):
                made_switch = False
                score = self._score_tree_alignment(comp)
                for node in self.tree.get_nonterminals():
                    node.clades.reverse()
                    new_score = self._score_tree_alignment(comp)
                    if new_score > score:
                        made_switch = True
                    else:
                        node.clades.reverse()
                if not made_switch:
                    break

        def _score_tree_alignment(self, comp: 'Phylogeny'):
            self_leaf_order = self._leaf_order()
            comp_leaf_order = comp._leaf_order()
            shared = list(set(self_leaf_order.keys()) & set(comp_leaf_order.keys()))
            res = stats.spearmanr(
                [self_leaf_order[i] for i in shared],
                [comp_leaf_order[i] for i in shared]
            )
            return res.statistic

        def _leaf_order(self):
            return {
                node.name: i
                for i, node in enumerate(self.tree.get_terminals())
            }


        def compare(self, comp: 'Phylogeny'):
            concordance = self._calc_concordance(comp)

            # If there are fewer than 3 leafs, this cannot take place
            if concordance is None:
                return mo.md("Not enough shared genomes to compare.")

            # Align the two trees against each other
            self.align_trees(comp)
            comp.align_trees(self)

            # Regenerate the coordinates
            self.find_coords()

            # Plot the two trees against each other

            # Set up a figure with two subplots
            fig = make_subplots(
                cols=3,
                rows=1,
                subplot_titles=(self.name, None, comp.name),
                shared_yaxes=True,
                horizontal_spacing=0.,
                vertical_spacing=0.,
                column_widths=[2, 1, 2]
            )

            shared_nodes = list(set(self._get_leafs(self.tree)) & set(comp._get_leafs(comp.tree)))

            self._plot_lines(fig)
            self._plot_points(fig, mode="markers")
            self._plot_tracer(fig, shared_nodes)

            comp._plot_lines(fig, row=1, col=3)
            comp._plot_points(fig, mode="markers", row=1, col=3)
            comp._plot_tracer(fig, shared_nodes, row=1, col=3)

            # Draw lines between each shared leaf
            for node_name in shared_nodes:
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[self.coords[node_name]['y'], comp.coords[node_name]['y']],
                        mode="lines",
                        showlegend=False,
                        line=dict(dash='dot', color="gray"),
                        cliponaxis=False
                    ),
                    row=1,
                    col=2
                )

            blank_axis = dict(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False
            )

            fig.update_layout(
                template="simple_white",
                yaxis=blank_axis,
                yaxis2=blank_axis,
                yaxis3=blank_axis,
                xaxis=dict(
                    automargin=True,
                    title_text="SNP Rate"
                ),
                xaxis2=blank_axis,
                xaxis3=dict(
                    automargin=True,
                    title_text="SNP Rate",
                    autorange="reversed"
                ),
                margin=dict(l=100, r=400, b=100, t=100),
                title_text=f"{self.name} vs. {comp.name} (Pairwise Distances - Spearman rho: {concordance:.1f})"
            )

            return mo.ui.plotly(fig)


        def _calc_concordance(self, comp: 'Phylogeny'):
            """
            Concordance: Spearman correlation of distances for all shared nodes.
            Nodes are shared if both trees contain a node with the same set of leafs.
            """
            # Get the shared set of leafs for both trees
            shared_leafs = list(set(self._get_leafs(self.tree)) & set(self._get_leafs(comp.tree)))
            # If there are fewer than 3 shared leafs, return null
            if len(shared_leafs) < 3:
                return

            # Get the vector of pairwise distances for this bin
            dists1 = [
                self.distances[name1][name2]
                for name1 in shared_leafs
                for name2 in shared_leafs
                if name1 < name2
            ]
            # And the comparitor
            dists2 = [
                comp.distances[name1][name2]
                for name1 in shared_leafs
                for name2 in shared_leafs
                if name1 < name2
            ]

            # Calculate the spearman correlation
            r = stats.spearmanr(dists1, dists2)
            return r.statistic

        def _get_leafs(self, node: Tree):
            return [leaf.name for leaf in node.get_terminals()]

        def _get_node_terminals(self, tree: Tree, shared_leafs: set):
            nodes = [
                set(self._get_leafs(node))
                for node in tree.get_nonterminals()
                if len(node.get_terminals()) > 1
            ]

            # Only keep the shared leafs (genomes)
            nodes = [
                frozenset(node & shared_leafs)
                for node in nodes
                if len(node & shared_leafs) > 1
            ]

            return set(nodes)
    return (Phylogeny,)


@app.cell(hide_code=True)
def _(
    AnnData,
    DataPortalAssetNotFound,
    DataPortalDataset,
    Dict,
    List,
    Phylogeny,
    Tuple,
    cluster,
    defaultdict,
    lru_cache,
    mo,
    np,
    pd,
    query_params,
    read_csv_cached,
    read_text_cached,
    spatial,
):
    # Define an object with all of the information for a pangenome
    class Pangenome:
        """
        Attributes:
            ds: DataPortalDataset
            bin_contents: Dict[str, pd.DataFrame]
            adata: AnnData
                obs: genomes
                var: bins
        """

        ds: DataPortalDataset
        adata: AnnData
        bin_contents: Dict[str, pd.DataFrame]
        genome_tree: cluster.hierarchy.ClusterNode
        tree_df: pd.DataFrame
        clades: List[set]

        def __init__(self, ds: DataPortalDataset, min_prop: float):
            self.ds = ds

            # Read in the table of which genes are in which bins
            # and turn into a dict keyed by bin ID
            self.bin_contents = {
                bin: d.drop(columns=['bin'])
                for bin, d in self.read_csv(
                    "data/bin_pangenome/gene_bins.csv",
                    low_memory=False
                ).groupby("bin")
            }

            # Build an AnnData object
            self.adata = self._make_adata()

            # Make a binary mask layer indicating whether each genome meets the min_prop threshold
            self.adata.layers["present"] = (self.adata.to_df() >= min_prop).astype(int)

            # Calculate the number of genes detected per genome
            # (counting each entire bin that was detected per genome)
            self.adata.obs["n_genes"] = pd.Series({
                genome: self.adata.var.loc[genome_contains_bin, "n_genes"].sum()
                for genome, genome_contains_bin in (self.adata.to_df(layer="present") == 1).iterrows()
            })

            # Filter out the genomes which are below the threshold
            _n_genomes_filter_threshold = (
                self.adata.obs['n_genes'].median()
                *
                float(query_params.get("min_genes_rel_median", '0.5'))
            )
            _n_genomes_filter = (self.adata.obs['n_genes'] >= _n_genomes_filter_threshold)
            self.n_genomes_filtered_out = self.adata.shape[0] - _n_genomes_filter.sum()
            self.adata = self.adata[_n_genomes_filter]

            # Filter the dataset to only include genomes which include a gene
            self.adata = self.adata[self.adata.obs["n_genes"] > 0]

            # Compute the number of genomes that each bin is found in
            self.adata.var["n_genomes"] = self.adata.to_df(layer="present").sum()

            # Filter out any bins which are found in 0 genomes
            self.adata = self.adata[:, self.adata.var["n_genomes"] > 0]

            # Compute the proportion of genomes that each bin is found in
            self.adata.var["prevalence"] = self.adata.var["n_genomes"] / self.adata.shape[0]

            # Run linkage clustering on the genomes
            self._genome_tree, self._genome_tree_nodes = cluster.hierarchy.to_tree(
                cluster.hierarchy.linkage(
                    spatial.distance.squareform(
                        self.adata.obsp["ani_distance"]
                    ),
                    method="average"
                ),
                rd=True
            )

            # Get all of the possible monophyletic groupings of genomes, based on that binary tree
            self.index_tree()

            # Compute the monophyly score for each bin, and then summarize across each genome
            self.compute_monophyly()

            # Read the table listing coordinates in each genome where the genes aligned
            self.genome_aln_coords = {
                sseqid: _genome_coords
                for sseqid, _genome_coords in self.read_csv("data/align/genomes.aln.csv.gz").groupby("sseqid")
            }

        def compute_monophyly(self):
            """
            Compute the monophyly score for each bin.
            (# of genomes the bin is present in) / 
            (# of genomes in the smallest monophyletic clade containing those genomes)
            """

            # Update the clades to only include what is present in the filtered dataset
            self.clades = [
                clade & set(self.adata.obs_names.values)
                for clade in self.clades
            ]

            # Annotate each bin by its monophyly score
            self.adata.var["monophyly"] = pd.Series({
                bin: self._compute_monophyly(bin)
                for bin in self.adata.var_names
            })

            # Annotate each genome by the weighted average of the monophyly score for each
            # bin it contains (weighted by the bin size)
            self.adata.obs["monophyly"] = pd.Series({
                genome: (
                    self.adata.var
                    .loc[genome_contains_bin]
                    .apply(
                        lambda r: r['monophyly'] * r['n_genes'],
                        axis=1
                    )
                    .sum()
                    / self.adata.obs.loc[genome, "n_genes"]
                )
                for genome, genome_contains_bin in (self.adata.to_df(layer="present") == 1).iterrows()
            })

        def _compute_monophyly(self, bin: str):

            # Get the set of genomes which contain this bin
            genomes = set(self.adata.obs_names.values[self.adata.to_df(layer="present")[bin] == 1])
            assert len(genomes) == self.adata.to_df(layer="present")[bin].sum(), bin

            # If there are none, stop
            if len(genomes) == 0:
                return

            # If there is only one, return 1
            elif len(genomes) == 1:
                return 1

            # If there are more, start by finding the smallest clade which contains all of the genomes
            else:
                smallest_clade = np.min([
                    len(clade)
                    for clade in self.clades
                    if genomes <= clade and len(clade) > 0
                ])
                # assert len(genomes) < 3, (genomes, smallest_clade)
                # Return the proportion of genomes in that clade which contain the bin
                return len(genomes) / smallest_clade

        def index_tree(self):
            """Get all of the possible monophyletic groupings of genomes, based on that binary tree."""

            self.clades = [
                set([
                    self.adata.obs_names[ix]
                    for ix in cn.pre_order(lambda n: n.id)
                    if ix < self.adata.obs.shape[0]
                ])
                for cn in self._genome_tree_nodes
                if cn.get_count() > 1
            ]

        def _make_adata(self) -> AnnData:

            # The variable annotations includes the number of genes for each variable
            var = pd.DataFrame(dict(
                n_genes=pd.Series({bin: d.shape[0] for bin, d in self.bin_contents.items()})
            ))

            # Read in the table listing which genomes contain which bins
            _genome_contents = self.read_csv(
                "data/bin_pangenome/genome_content.long.csv",
                low_memory=False
            )
            if _genome_contents.shape[0] == 0:
                raise Exception("Error: Missing contents in genome_content.long.csv")

            # Split off the genome annotation information into a separate table
            obs = (
                _genome_contents
                .drop(columns=["bin", "n_genes_detected", "prop_genes_detected"])
                .drop_duplicates()
                .groupby("genome")
                .head(1)
                .set_index("genome")
                .fillna("None")
            )

            # X will be the proportion of genes detected
            X = (
                _genome_contents
                .sort_values(by="prop_genes_detected", ascending=False)
                .groupby(["genome", "bin"])
                .head(1)
                .pivot(
                    index="genome",
                    columns="bin",
                    values="prop_genes_detected"
                )
                .fillna(0)
            )
            print(f"Genomes x Bins Detected: {X.shape[0]:,} x {X.shape[1]:,}")

            if X.shape[0] == 0 or X.shape[1] == 0:
                raise ValueError("No data found for this pangenome.")

            # Read in the ANI distance matrix for all genomes
            _genome_ani = self.read_csv(
                "data/distances.csv.gz",
                index_col=0
            )
            print("Building AnnData object")

            # Build an AnnData object
            return AnnData(
                X=X,
                obs=obs.reindex(index=X.index),
                var=var.reindex(index=X.columns),
                obsp=dict(
                    ani_distance=_genome_ani.reindex(
                        columns=X.index,
                        index=X.index
                    )
                )
            )

        def read_csv(self, fp: str, **kwargs):
            return read_csv_cached(
                self.ds.project_id,
                self.ds.id,
                fp,
                **kwargs
            )

        def read_fasta(self, fp: str, **kwargs):
            fasta = {}
            header = None
            sequence = []

            for line in read_text_cached(
                self.ds.project_id,
                self.ds.id,
                fp,
                **kwargs
            ).split("\n"):
                if len(line) == 0:
                    pass
                elif line[0] == '>':
                    if header is not None:
                        fasta[header] = ''.join(sequence)
                    header = line[1:]
                    sequence = []
                else:
                    sequence.append(line)

            if header is not None:
                fasta[header] = ''.join(sequence)
            return fasta

        @lru_cache
        def bin_phylogeny(self, bin_id: str, n_genes: int) -> Phylogeny:
            phy = self._read_phylogeny(bin_id, n_genes)
            return phy

        def _read_phylogeny(self, bin_id: str, n_genes: int):
            # Get the pairwise proportion of SNPs for every pair of genomes, across every gene
            snp_rate = self._merge_gene_snps({
                gene_id: self._read_gene_snps(gene_id)
                for gene_id in (
                    self.bin_contents[bin_id]
                    .sort_values(by="n_genomes", ascending=False)
                    .head(n_genes)
                    ["gene_id"]
                )
            })

            return Phylogeny(bin_id, snp_rate)

        def _merge_gene_snps(self, snps: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
            all_genomes = list(set([
                genome_id
                for df, _ in snps.values()
                if isinstance(df, pd.DataFrame)
                for genome_id in df.columns.values
            ]))
            ix_kwargs = dict(
                index=all_genomes,
                columns=all_genomes
            )
            all_snps = pd.DataFrame(
                np.zeros((len(all_genomes), len(all_genomes))),
                **ix_kwargs
            )
            all_overlap = pd.DataFrame(
                np.zeros((len(all_genomes), len(all_genomes))),
                **ix_kwargs
            )

            # Add up the pairwise identity data for each gene
            for overlap_df, snp_df in snps.values():
                if overlap_df is not None:
                    all_overlap = all_overlap + overlap_df.reindex(**ix_kwargs).fillna(0)
                    all_snps = all_snps + snp_df.reindex(**ix_kwargs).fillna(0)

            # Calculate the SNP rate
            snp_rate = (all_snps.fillna(0) / all_overlap.fillna(0)).fillna(0)

            return snp_rate

        def _read_gene_snps(self, gene_id: str):
            # Read in the FASTA with the aligned nucleotide sequences
            try:
                gene_fasta = self.read_fasta(f"data/align/genes/{gene_id}.fasta.gz", compression="gzip")
            except DataPortalAssetNotFound:
                return None, None

            # For each sequence, pad the beginning if the gene doesn't align from the first position
            # Use the genome alignment table to find when the alignment was partial at the beginning
            coords = self.genome_aln_coords[gene_id].set_index("genome")
            # Padded sequences
            seqs = {
                genome_id: str(np.char.multiply("---", coords.loc[genome_id, "sstart"] - 1)) + aln_seq
                for genome_id, aln_seq in gene_fasta.items()
                if genome_id in coords.index
            }

            # For each pair of genomes, count up the number of shared positions, and the number of SNPs
            overlap, snps = self._calc_snps(seqs)

            return overlap, snps

        def _calc_snps(self, seqs: Dict[str, str]):
            overlap = defaultdict(lambda: defaultdict(int))
            snps = defaultdict(lambda: defaultdict(int))
            for genome1, seq1 in seqs.items():
                for genome2, seq2 in seqs.items():
                    if genome1 < genome2:
                        for i in range(min(len(seq1), len(seq2))):
                            if seq1[i] in ["A", "T", "C", "G"] and seq2[i] in ["A", "T", "C", "G"]:
                                overlap[genome1][genome2] += 1
                                overlap[genome2][genome1] += 1
                                if seq1[i] != seq2[i]:
                                    snps[genome1][genome2] += 1
                                    snps[genome2][genome1] += 1

            genomes = list(seqs.keys())
            return (
                pd.DataFrame(overlap).reindex(index=genomes, columns=genomes).fillna(0).astype(int),
                pd.DataFrame(snps).reindex(index=genomes, columns=genomes).fillna(0).astype(int)
            )


    # Cache the creation of pangenome objects
    def make_pangenome(ds: DataPortalDataset, min_prop: float):
        with mo.status.spinner("Loading data..."):
            return Pangenome(ds, min_prop)
    return Pangenome, make_pangenome


@app.cell(hide_code=True)
def _(mo):
    with mo.status.spinner("Loading dependencies:"):
        import plotly.express as px
        from plotly.subplots import make_subplots
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    return Patch, make_subplots, np, plt, px, sns


@app.cell
def _(
    DataPortalDataset,
    Pangenome,
    Tuple,
    cluster,
    go,
    make_pangenome,
    make_subplots,
    mo,
    np,
    pd,
    px,
    query_params,
    sklearn,
):
    # Object used to configure and display the "Inspect Pangenome" section
    class InspectPangenome:
        pg: Pangenome

        def __init__(self, ds: DataPortalDataset):
            self.pg = make_pangenome(ds, float(query_params.get("min_prop", 0.5)))

            # Precompute the coordinates for the genomes
            # Ordinate using the log-transformed gene abundances, to not overrepresent large bins
            log_genes_per_genome = self.pg.adata.to_df(layer="present") * self.pg.adata.var["n_genes"].apply(np.log10)
            self.genome_coords = sklearn.manifold.Isomap().fit_transform(log_genes_per_genome)

            # Make a table with the number of genes from each bin which are found in each genome
            self.n_genes_per_genome = self.pg.adata.to_df(layer="present") * self.pg.adata.var["n_genes"]

            # Calculate the proportion of each genome made up by each bin
            self.prop_genes_per_genome = self.n_genes_per_genome.apply(lambda r: r / r.sum(), axis=1)

        def plot_heatmap_args(self):
            return mo.md("""
    ### Inspect Pangenome: Heatmap

     - {top_n_bins}
     - {include_bins}
     - {top_n_genomes}
     - {include_genomes}
     - {height}
            """).batch(
                top_n_bins=mo.ui.number(
                    label="Show N Bins (Largest # of Genes):",
                    start=0,
                    value=40,
                    step=1
                ),
                include_bins=mo.ui.multiselect(
                    label="Show Specific Bins:",
                    value=[],
                    options=self.pg.adata.var_names
                ),
                top_n_genomes=mo.ui.number(
                    label="Show N Genomes (Largest # of Genes):",
                    start=0,
                    value=200,
                    step=1
                ),
                include_genomes=mo.ui.multiselect(
                    label="Show Specific Genomes:",
                    value=[],
                    options=self.pg.adata.obs_names
                ),
                height=mo.ui.number(
                    label="Figure Height",
                    start=100,
                    value=800
                )
            )

        def plot_heatmap(
            self,
            top_n_bins: int,
            include_bins: list,
            top_n_genomes: int,
            include_genomes: list,
            height: int,
        ):
            """Make a heatmap showing which bins are present in which genomes."""

            # First get the top bins selected based on bin size
            if top_n_bins is not None and top_n_bins > 0:
                bins_to_plot = list(self.pg.adata.var.sort_values(by="n_genes", ascending=False).head(top_n_bins).index.values)
            else:
                bins_to_plot = []

            for bin in include_bins:
                if bin not in bins_to_plot:
                    bins_to_plot.append(bin)

            if len(bins_to_plot) < 2:
                return mo.md("""Please select multiple bins to plot""")

            # Next get the top genomes selected based on genome size
            if top_n_genomes is not None and top_n_genomes > 0:
                genomes_to_plot = list(
                    self.n_genes_per_genome
                    .sum(axis=1)
                    .sort_values(ascending=False)
                    .head(top_n_genomes)
                    .index.values
                )
            else:
                genomes_to_plot = []

            for genome in include_genomes:
                if genome not in genomes_to_plot:
                    genomes_to_plot.append(genome)

            if len(genomes_to_plot) < 2:
                return mo.md("""Please select multiple genomes to plot""")

            present = (
                self.pg.adata
                [genomes_to_plot, bins_to_plot]
                .to_df(layer="present")
            )

            genome_order = sort_axis(present)
            bin_order = sort_axis(present.T)

            fig = make_subplots(
                shared_xaxes=True,
                shared_yaxes=True,
                rows=4,
                cols=3,
                start_cell="bottom-left",
                column_widths=[2, 1, 1],
                row_heights=[2, 1, 1, 1],
                horizontal_spacing=0.01,
                vertical_spacing=0.01
            )
            fig.add_heatmap(
                x=bin_order,
                y=genome_order,
                z=present.reindex(index=genome_order, columns=bin_order),
                text=[
                    [
                        f"{genome}<br>{bin}<br>{'Present' if present.loc[genome, bin] else 'Absent'}<extra></extra>"
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
                y=self.pg.adata.var["n_genes"].reindex(bin_order),
                x=bin_order,
                row=2,
                col=1,
                showlegend=False,
                text=[
                    f"{bin_id}<br>{self.pg.adata.var.loc[bin_id, 'n_genes']:,} Genes"
                    for bin_id in bin_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.add_bar(
                x=self.pg.adata.obs["n_genes"].reindex(genome_order),
                y=genome_order,
                orientation='h',
                row=1,
                col=2,
                showlegend=False,
                text=[
                    f"{genome_id}<br>{self.pg.adata.obs.loc[genome_id, 'n_genes']:,} Genes"
                    for genome_id in genome_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.add_bar(
                x=self.pg.adata.obs["monophyly"].reindex(genome_order),
                y=genome_order,
                orientation='h',
                row=1,
                col=3,
                showlegend=False,
                text=[
                    f"{genome_id}<br>{self.pg.adata.obs.loc[genome_id, 'monophyly']:,} Monophyly Score"
                    for genome_id in genome_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.add_bar(
                y=self.pg.adata.var["monophyly"].reindex(bin_order),
                x=bin_order,
                row=3,
                col=1,
                showlegend=False,
                text=[
                    f"{bin_id}<br>{self.pg.adata.var.loc[bin_id, 'monophyly']:,} Monophyly Score"
                    for bin_id in bin_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.add_bar(
                y=self.pg.adata.var["prevalence"].reindex(bin_order),
                x=bin_order,
                row=4,
                col=1,
                showlegend=False,
                text=[
                    f"{bin_id}<br>{self.pg.adata.var.loc[bin_id, 'prevalence']:,} Prevalence"
                    for bin_id in bin_order
                ],
                hovertemplate="%{text}<extra></extra>"
            )
            fig.update_layout(
                height=height,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                bargap=0,
                yaxis4=dict(
                    title_text="Bin Size<br>(# of Genes)",
                    type="log",
                ),
                yaxis7=dict(title_text="Monophyly<br>Score"),
                yaxis10=dict(title_text="Prevalence"),
                yaxis=dict(
                    # side="right",
                    title_text="Genome<br>Presence / Absence",
                    showticklabels=False
                ),
                xaxis2=dict(
                    title_text="Genome Size (# of Genes)"
                ),
                xaxis3=dict(
                    title_text="Monophyly Score"
                )
            )
            return fig

        def summary_metrics(self):
            return mo.accordion({
                "Pangenome Characteristics": mo.md(f"""
                | Pangenome Characteristics | |
                | --- | --- |
                | Number of Genomes | {self.pg.adata.n_obs:,} |
                | Number of Genes | {self.pg.adata.var["n_genes"].sum():,} |
                | Number of Gene Bins | {self.pg.adata.n_vars:,} |
                | Genome Size (max) | {self.pg.adata.obs["n_genes"].max()} |
                | Genome Size (75%) | {int(self.pg.adata.obs["n_genes"].quantile(0.75))} |
                | Genome Size (50%) | {int(self.pg.adata.obs["n_genes"].quantile(0.5))} |
                | Genome Size (25%) | {int(self.pg.adata.obs["n_genes"].quantile(0.25))} |
                | Genome Size (min) | {self.pg.adata.obs["n_genes"].min()} |
                | Genomes Omitted: Low # of Genes | {self.pg.n_genomes_filtered_out:,} |
                """)
            })

        def plot_scatter_args(self):
            return mo.md("""
    ### Inspect Pangenome: Genome Map

    - Show N Bins: {n_bins}
    - {top_n_genomes}
    - {include_genomes}
    - Annotate Genomes By: {annotate_by}
    - Annotation Offset (x): {label_offset_x}
    - Annotation Offset (y): {label_offset_y}
    - Include Specific Bins: {include_bins}
    - {size_max}
    - {height}
            """).batch(
                annotate_by=mo.ui.dropdown(
                    options=self.pg.adata.obs.reset_index().columns.values,
                    value=query_params.get(
                        "inspect_genomes_annotate_by",
                        (
                            "averageNucleotideIdentity_bestAniMatch_organismName"
                            if "averageNucleotideIdentity_bestAniMatch_organismName" in self.pg.adata.obs_keys()
                            else self.pg.adata.obs.reset_index().columns.values[0]
                        )
                    ),
                    searchable=True
                ),
                label_offset_x=mo.ui.number(value=0.),
                label_offset_y=mo.ui.number(value=0.5),
                n_bins=mo.ui.number(
                    start=0,
                    stop=100,
                    step=1,
                    value=10
                ),
                top_n_genomes=mo.ui.number(
                    label="Show N Genomes (Largest # of Genes):",
                    start=0,
                    value=200,
                    step=1
                ),
                include_genomes=mo.ui.multiselect(
                    label="Show Specific Genomes:",
                    value=[],
                    options=self.pg.adata.obs_names
                ),
                include_bins=mo.ui.multiselect(
                    options=self.pg.adata.var_names,
                    value=[]
                ),
                height=mo.ui.number(
                    label="Figure Height",
                    start=100,
                    value=600
                ),
                size_max=mo.ui.number(
                    label="Point Size:",
                    start=1,
                    value=20
                )
            )

        def plot_scatter(
            self,
            annotate_by: str,
            n_bins: int,
            top_n_genomes: int,
            include_genomes: list,
            include_bins: list,
            height: int,
            label_offset_x: float,
            label_offset_y: float,
            size_max: int
        ):
            """
            Show a scatter plot where each point is a genome.
            The layout is driven by the similarity of which genes are encoded in the genome.
            Annotations can be made according to the presence of different bins, genome size, etc.
            """
            if n_bins == 0 and len(include_bins) == 0:
                return mo.md("Select bins to display")

            with mo.status.spinner("Generating Plot:"):

                # Choose the bins to plot based on the mean proportion across genomes
                bins_to_plot = list(self.prop_genes_per_genome.mean().sort_values().tail(n_bins).index.values)

                # Add any specific bins selected by the user
                for _bin in include_bins:
                    if _bin not in bins_to_plot:
                        bins_to_plot.append(_bin)

                # Next get the top genomes selected based on genome size
                if top_n_genomes is not None and top_n_genomes > 0:
                    genomes_to_plot = list(
                        self.n_genes_per_genome
                        .sum(axis=1)
                        .sort_values(ascending=False)
                        .head(top_n_genomes)
                        .index.values
                    )
                else:
                    genomes_to_plot = []

                for genome in include_genomes:
                    if genome not in genomes_to_plot:
                        genomes_to_plot.append(genome)

                # Drop all of the unselected bins, and add an "Other" column
                n_genes_per_genome = pd.concat([
                    self.n_genes_per_genome.reindex(
                        columns=bins_to_plot,
                        index=genomes_to_plot,
                    ),
                    pd.DataFrame(dict(Other=self.n_genes_per_genome.drop(columns=bins_to_plot).sum(axis=1)))
                ], axis=1)

                # Make a label for each genome
                genome_label = {
                    genome: "<br>".join([
                        f"{bin}:&#9;{n_genes:,} genes"
                        for bin, n_genes
                        in genome_count.sort_values(ascending=False).items()
                        if n_genes > 0
                    ])
                    for genome, genome_count in n_genes_per_genome.iterrows()
                }

                # Make a table to use only for point size
                # For each genome, take the cumulative sum from smallest to largest
                point_size_per_genome = (
                    n_genes_per_genome
                    .apply(lambda r: (r > 0).apply(int), axis=1)
                    .apply(lambda r: r.iloc[::-1].cumsum(), axis=1)
                    .apply(lambda r: r / r.max(), axis=1)
                )

                # Make a table to plot which includes the proportion for every single bin
                plot_df = pd.DataFrame([
                    dict(
                        genome=genome,
                        x=genome_coord[0],
                        y=genome_coord[1],
                        bin=bin,
                        genes_in_bin=n_genes_per_genome.loc[genome, bin],
                        point_size=point_size_per_genome.loc[genome, bin],
                        text=genome_label[genome]
                    )
                    for genome, genome_coord in zip(self.pg.adata.obs_names, self.genome_coords)
                    if genome in genomes_to_plot
                    for bin in point_size_per_genome.columns.values
                    if n_genes_per_genome.loc[genome, bin] > 0
                ]).sort_values(by="point_size", ascending=False)

                # Add the genome metadata
                plot_df = plot_df.merge(
                    self.pg.adata.obs,
                    right_index=True,
                    left_on="genome"
                )

                fig = px.scatter(
                    plot_df,
                    x="x",
                    y="y",
                    custom_data=[annotate_by, "text"],
                    hover_data=["genes_in_bin"],
                    size="point_size",
                    color="bin",
                    hover_name=annotate_by,
                    template="simple_white",
                    opacity=1,
                    labels=dict(bin="Gene Bin"),
                    size_max=size_max
                )
                text_df = pd.DataFrame(dict(
                    x=self.genome_coords[:, 0] + label_offset_x,
                    y=self.genome_coords[:, 1] + label_offset_y,
                    text=[
                        (
                            genome if annotate_by == "genome"
                            else
                            self.pg.adata.obs.loc[genome, annotate_by]
                        )
                        for genome in self.pg.adata.obs_names
                    ],
                    genome=self.pg.adata.obs_names
                ))
                text_df = text_df.loc[text_df["genome"].isin(genomes_to_plot)]
                fig.add_scatter(
                    x=text_df["x"],
                    y=text_df["y"],
                    text=text_df["text"],
                    mode="text",
                    legendgroup="Annotation",
                    legendgrouptitle=dict(text="Annotation"),
                    name=annotate_by
                )

                fig.update_traces(
                    hovertemplate="<b>%{customdata[0]}</b><br><br>%{customdata[1]}<extra></extra>",
                    marker=dict(line=dict(width=0))
                )
                fig.update_layout(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    height=height
                )
                return fig



    def sort_axis(df: pd.DataFrame, metric="jaccard", method="average"):
        return df.index.values[
            cluster.hierarchy.leaves_list(
                cluster.hierarchy.linkage(
                    df.values,
                    method=method,
                    metric=metric
                )
            )
        ]
    return InspectPangenome, sort_axis


@app.cell
def _(InspectPangenome, pangenome_dataset):
    # Initialize the object using the dataset selected by the user
    inspect_pangenome = InspectPangenome(pangenome_dataset)
    return (inspect_pangenome,)


@app.cell
def _(inspect_pangenome):
    inspect_pangenome.summary_metrics()
    return


@app.cell
def _(inspect_pangenome):
    inspect_pangenome_heatmap_args = inspect_pangenome.plot_heatmap_args()
    inspect_pangenome_heatmap_args
    return (inspect_pangenome_heatmap_args,)


@app.cell
def _(inspect_pangenome, inspect_pangenome_heatmap_args):
    inspect_pangenome.plot_heatmap(**inspect_pangenome_heatmap_args.value)
    return


@app.cell
def _(inspect_pangenome, mo):
    mo.download(
        inspect_pangenome.pg.adata.var.to_csv(),
        label="Pangenome Bin Summary Table",
        filename=f"{inspect_pangenome.pg.ds.name} - Bin Summary Table.csv"
    )
    return


@app.cell
def _(inspect_pangenome, mo):
    mo.download(
        inspect_pangenome.pg.adata.obs.to_csv(),
        label="Pangenome Genome Summary Table",
        filename=f"{inspect_pangenome.pg.ds.name} - Genome Summary Table.csv"
    )
    return


@app.cell
def _(inspect_pangenome):
    inspect_pangenome_plot_scatter_args = inspect_pangenome.plot_scatter_args()
    inspect_pangenome_plot_scatter_args
    return (inspect_pangenome_plot_scatter_args,)


@app.cell
def _(inspect_pangenome, inspect_pangenome_plot_scatter_args):
    inspect_pangenome.plot_scatter(**inspect_pangenome_plot_scatter_args.value)
    return


@app.cell
def _(mo):
    mo.md(r"""## Inspect Gene Bin""")
    return


@app.cell
def _(DataPortalDataset, Pangenome, make_pangenome, mo, query_params):
    class InspectGeneBin:
        pg: Pangenome

        def __init__(self, ds: DataPortalDataset):
            self.pg = make_pangenome(ds, float(query_params.get("min_prop", 0.5)))

        def display_bin_args(self):
            return mo.md(
                """
        {bin_id}

        {genome_columns}
                """
            ).batch(
                bin_id=mo.ui.dropdown(
                    label="Display:",
                    options=sorted(self.pg.adata.var_names, key=lambda bin_id: int(bin_id.split(" ")[-1])),
                    value="Bin 1",
                    searchable=True
                ),
                genome_columns=mo.ui.multiselect(
                    label="Show Columns:",
                    options=self.pg.adata.obs.columns.values,
                    value=self.pg.adata.obs.columns.values
                )
            )

        def display_bin(self, bin_id, genome_columns):
            # Show the list of genes in the bin
            gene_df = (
                self.pg.bin_contents[bin_id]
                .set_index("gene_id")
                .rename(
                    columns=dict(
                        combined_name="Gene Annotation",
                        n_genomes="Number of Genomes"
                    )
                )
            )

            # Show the list of genomes containing the bin
            genome_df = (
                self.pg.adata
                [
                    self.pg.adata
                    .to_df(layer="present")
                    [bin_id] == 1
                ]
                .obs
            )

            return mo.vstack([
                mo.md(bin_id),
                mo.md(f"{gene_df.shape[0]:,} Genes / {genome_df.shape[0]:,} Genomes"),
                gene_df,
                genome_df.reindex(columns=genome_columns)
            ])

        def display_bin_layout(self, bin_id, **kwargs):
            # Show the layout of the bin as an iFrame
            try:
                html = (
                    self.pg
                    .ds
                    .list_files()
                    .get_by_name(f"data/bin_pangenome/layout/{bin_id}.html")
                    .read()
                )
            except Exception as e:
                return mo.md(f"No gene layout available for {bin_id}")
            filename = f"{self.pg.ds.name} {bin_id} Layout.html"
            return mo.download(
                html,
                label=f"Download {filename}",
                filename=filename,
                mimetype="text/html"
            )

        def display_bin_phylogeny_args(self, bin_id, **kwargs):
            """Options from the user needed to display a phylogeny."""
            return mo.md("""
    ### Display Phylogeny

    - {n_genes}
            """).batch(
                n_genes=mo.ui.number(
                    label="Sample N Genes:",
                    start=1,
                    step=1,
                    value=10
                )
            )

        def display_bin_phylogeny(self, bin_id: str, n_genes: int, **kwargs):
            """Show the phylogeny of a bin."""

            # Read the phylogeny of a bin
            phy = self.pg.bin_phylogeny(bin_id, n_genes)

            return phy.plot()


    return (InspectGeneBin,)


@app.cell
def _(InspectGeneBin, pangenome_dataset):
    inspect_gene_bin = InspectGeneBin(pangenome_dataset)
    return (inspect_gene_bin,)


@app.cell
def _(inspect_gene_bin):
    # Show the user information about a specific bin
    display_bin_args = inspect_gene_bin.display_bin_args()
    display_bin_args
    return (display_bin_args,)


@app.cell
def _(display_bin_args, inspect_gene_bin):
    inspect_gene_bin.display_bin(**display_bin_args.value)
    return


@app.cell
def _(display_bin_args, inspect_gene_bin):
    inspect_gene_bin.display_bin_layout(**display_bin_args.value)
    return


@app.cell
def _(display_bin_args, inspect_gene_bin):
    display_bin_phylogeny_args = inspect_gene_bin.display_bin_phylogeny_args(**display_bin_args.value)
    display_bin_phylogeny_args
    return (display_bin_phylogeny_args,)


@app.cell
def _(display_bin_args, display_bin_phylogeny_args, mo):
    run_phylogeny_button = mo.ui.run_button(label=f"Compute Phylogeny for {display_bin_args.value['bin_id']} (read {display_bin_phylogeny_args.value['n_genes']:,} gene alignments)")
    run_phylogeny_button
    return (run_phylogeny_button,)


@app.cell
def _(
    display_bin_args,
    display_bin_phylogeny_args,
    inspect_gene_bin,
    mo,
    run_phylogeny_button,
):
    mo.stop(not run_phylogeny_button.value)

    inspect_gene_bin.display_bin_phylogeny(
        **display_bin_phylogeny_args.value,
        **display_bin_args.value
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Compare Gene Bins""")
    return


@app.cell
def _(
    DataPortalDataset,
    List,
    Pangenome,
    make_pangenome,
    make_subplots,
    mo,
    pd,
    query_params,
):
    class CompareGeneBins:
        pg: Pangenome

        def __init__(self, ds: DataPortalDataset):
            self.pg = make_pangenome(ds, float(query_params.get("min_prop", 0.5)))

        def args(self):
            return mo.md("""
    {bins}

    {genome_annot}

    {height}

            """).batch(
                bins=mo.ui.multiselect(
                    label="Bins:",
                    options=sorted(self.pg.adata.var_names, key=lambda bin_id: int(bin_id.split(" ")[-1])),
                    value=list(self.pg.adata.var.sort_values(by="n_genes", ascending=False).head(10).index.values)
                ),
                genome_annot=mo.ui.dropdown(
                    label="Genome Annotation:",
                    options=['None'] + self.pg.adata.obs_keys(),
                    value=(
                        "averageNucleotideIdentity_bestAniMatch_organismName"
                        if "averageNucleotideIdentity_bestAniMatch_organismName" in self.pg.adata.obs_keys()
                        else 'None'
                    ),
                    searchable=True
                ),
                height=mo.ui.number(
                    label="Figure Height:",
                    start=100,
                    value=800
                )
            )

        def show_bin_overlap(
            self,
            bins: list,
            height: int,
            genome_annot: str,
            genome_annot_groups=[]
        ):
            # Get the table of which genomes have these bins
            presence = (
                self
                .pg
                .adata
                .to_df(layer="present")
                .reindex(columns=bins)
                .groupby(bins)
                .apply(len, include_groups=False)
                .sort_values(ascending=False)
                .reset_index()
            )

            # Make a plot with two panels
            fig = make_subplots(
                shared_xaxes=True,
                rows=2 + int(genome_annot != 'None'),
                cols=1,
                start_cell="bottom-left",
                horizontal_spacing=0.01,
                vertical_spacing=0.1
            )

            # Make a long form table for the points
            points = pd.DataFrame([
                dict(
                    x=row_i,
                    y=bins.index(bin),
                    color="black" if present else "white",
                    bin=bin,
                    text=f"{bin} is {'Present' if present else 'Absent'}",
                    present=present
                )
                for row_i, row in presence.reindex(columns=bins).iterrows()
                for bin, present in row.items()
            ])

            # Add the vertical lines
            for x, x_df in points.query('present == 1').groupby("x"):
                if x_df.shape[0] > 1:
                    fig.add_scatter(
                        x=[x, x],
                        y=[x_df['y'].min(), x_df['y'].max()],
                        mode="lines",
                        line=dict(
                            color="grey"
                        ),
                        showlegend=False
                    )

            fig.add_scatter(
                x=points["x"],
                y=points["y"],
                mode="markers",
                marker=dict(
                    color=points["color"],
                    size=10,
                    line=dict(
                        width=1,
                        color="black"
                    )
                ),
                text=points["text"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False
            )

            fig.add_bar(
                x=list(range(presence.shape[0])),
                y=presence[0],
                row=2,
                col=1,
                showlegend=False,
                hovertemplate="%{y:,} Genomes<extra></extra>",
                marker=dict(color="black")
            )
            # Optionally add the annotation table
            if genome_annot != 'None':
                annot_df = (
                    pd.concat([
                        self.pg
                        .adata
                        .to_df(layer="present")
                        .reindex(columns=bins),
                        self.pg
                        .adata
                        .obs
                        .reindex(columns=[genome_annot])
                        .fillna('None')
                    ], axis=1)
                    .groupby(bins + [genome_annot])
                    .apply(len, include_groups=False)
                    .sort_values(ascending=False)
                    .reset_index()
                    .pivot_table(
                        columns=bins,
                        index=genome_annot,
                        values=0
                    )
                    .reindex(
                        columns=presence.set_index(bins).index
                    )
                    .fillna(0)
                )
                if len(genome_annot_groups) > 0:
                    annot_df = annot_df.reindex(index=genome_annot_groups)
                    annot_df = annot_df.dropna(
                        axis=0,
                        how="all"
                    )
                fig.add_heatmap(
                    z=annot_df.values,
                    y=annot_df.index.values,
                    row=3,
                    col=1,
                    colorscale="blues",
                    hovertemplate="%{y}<br>%{z} Genomes<extra></extra>"
                )
            fig.update_layout(
                template="simple_white",
                xaxis=dict(
                    showticklabels=False,
                    title_text="Presence / Absence"
                ),
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(bins))),
                    ticktext=bins
                ),
                yaxis2=dict(
                    title_text="Number of Genomes"
                ),
                height=height
            )
            return fig

        def compare_phylogeny_args(self, bins: List[str], **kwargs):
            """Options used to compare phylogenies of two bins."""
            if len(bins) <= 1:
                return mo.md("").batch()

            return mo.md("""
            ### Compare Bin Phylogenies
            - {n_genes}
            - {ref_bin}
            - {comp_bin}
            """).batch(
                n_genes=mo.ui.number(
                    start=1,
                    value=10,
                    label="Sample N Genes:"
                ),
                ref_bin=mo.ui.dropdown(
                    label="Reference Bin:",
                    options=bins,
                    value=bins[0],
                    searchable=True
                ),
                comp_bin=mo.ui.dropdown(
                    label="Comparison Bin:",
                    options=bins,
                    value=bins[1],
                    searchable=True
                )
            )

        def compare_phylogeny(self, n_genes: int, ref_bin: str, comp_bin: str):
            # Read the phylogeny of each bin
            with mo.status.spinner(f"Loading phylogeny for {ref_bin}"):
                ref_phy = self.pg.bin_phylogeny(ref_bin, n_genes)
            with mo.status.spinner(f"Loading phylogeny for {comp_bin}"):
                comp_phy = self.pg.bin_phylogeny(comp_bin, n_genes)

            return ref_phy.compare(comp_phy)

    return (CompareGeneBins,)


@app.cell
def _(CompareGeneBins, pangenome_dataset):
    compare_gene_bins = CompareGeneBins(pangenome_dataset)
    return (compare_gene_bins,)


@app.cell
def _(compare_gene_bins):
    # Show the user overlap between bins
    bin_overlap_args = compare_gene_bins.args()
    bin_overlap_args
    return (bin_overlap_args,)


@app.cell
def _(bin_overlap_args, compare_gene_bins, mo):
    # If an annotation is selected, let the user pick the groups
    def genome_annot_groups_options():
        return (
            []
            if bin_overlap_args.value["genome_annot"] == 'None'
            else
            compare_gene_bins.pg.adata.obs[bin_overlap_args.value["genome_annot"]].drop_duplicates().sort_values().values
        )

    genome_annot_groups = mo.ui.multiselect(
        label="Include Groups:",
        options=genome_annot_groups_options(),
        value=genome_annot_groups_options()
    )
    genome_annot_groups
    return (genome_annot_groups,)


@app.cell
def _(bin_overlap_args, compare_gene_bins, genome_annot_groups):
    compare_gene_bins.show_bin_overlap(
        genome_annot_groups=genome_annot_groups.value,
        **bin_overlap_args.value
    )
    return


@app.cell
def _(bin_overlap_args, compare_gene_bins, mo):
    # Options for comparing the phylogeny of two of these
    mo.stop(len(bin_overlap_args.value.get("bins", [])) < 2)
    compare_phylogeny_args = compare_gene_bins.compare_phylogeny_args(**bin_overlap_args.value)
    compare_phylogeny_args
    return (compare_phylogeny_args,)


@app.cell
def _(compare_phylogeny_args, mo):
    run_compare_phylogeny_button = mo.ui.run_button(label=f"Compare Phylogeny for {compare_phylogeny_args.value['ref_bin']} and {compare_phylogeny_args.value['comp_bin']} (read {compare_phylogeny_args.value['n_genes']:,} gene alignments each)")
    run_compare_phylogeny_button
    return (run_compare_phylogeny_button,)


@app.cell
def _(
    compare_gene_bins,
    compare_phylogeny_args,
    mo,
    run_compare_phylogeny_button,
):
    mo.stop(run_compare_phylogeny_button.value is False)
    compare_gene_bins.compare_phylogeny(**compare_phylogeny_args.value)
    return


@app.cell
def _(mo):
    mo.md(r"""## Inspect Metagenomes""")
    return


@app.cell
def _(client, filter_datasets_metagenome, mo, project_ui):
    # Stop if the user has not selected a project
    mo.stop(project_ui.value is None)

    # Get the list of metagenome datasets available to the user
    metagenome_datasets = [
        dataset
        for dataset in client.get_project_by_id(project_ui.value).list_datasets()
        if filter_datasets_metagenome(dataset)
    ]
    metagenome_datasets.sort(key=lambda ds: ds.created_at, reverse=True)
    return (metagenome_datasets,)


@app.cell
def _(id_to_name, metagenome_datasets, mo, name_to_id, query_params):
    # Let the user select which metagenome dataset to get data from
    inspect_metagenome_datasets_ui = mo.ui.multiselect(
        label="Select Metagenomes:",
        value=[
            id_to_name(
                metagenome_datasets,
                dataset_id
            )
            for dataset_id in (
                query_params.get("inspect_metagenome").split(",")
                if query_params.get("inspect_metagenome")
                else []
            )
        ],
        options=name_to_id(metagenome_datasets)
    )
    inspect_metagenome_datasets_ui
    return (inspect_metagenome_datasets_ui,)


@app.cell
def _(client, inspect_metagenome_datasets_ui, mo, project_ui):
    # Stop if the user has not selected a dataset
    mo.stop(inspect_metagenome_datasets_ui.value is None)

    # Get the selected dataset
    inspect_metagenome_datasets = [
        (
            client
            .get_project_by_id(project_ui.value)
            .get_dataset_by_id(dataset_id)
        )
        for dataset_id in inspect_metagenome_datasets_ui.value
    ]
    return (inspect_metagenome_datasets,)


@app.cell
def _():
    return


@app.cell
def _(inspect_metagenome_datasets, mo):
    mo.md("\n".join([
        f" - {dataset.name}"
        for dataset in inspect_metagenome_datasets
    ]))
    return


@app.cell
def define_metagenome_class(
    AnnData,
    DataPortalDataset,
    Dict,
    List,
    Pangenome,
    cluster,
    mo,
    np,
    pangenome_datasets,
    pd,
    query_params,
    read_csv_cached,
):
    class Metagenome:
        """
        Object containing information about a metagenome.
        Abundance information for each bin in each specimen is given as the number of aligned reads.
        To provide context for the abundance, each observation has metadata for:
            - genes:tot_reads
            - genes:aligned_reads
            - genes:detected_genes
        """
        ds: DataPortalDataset
        adata: AnnData
        bin_contents: Dict[str, pd.DataFrame]
        genome_tree: cluster.hierarchy.ClusterNode
        tree_df: pd.DataFrame
        clades: List[set]
        pg: Pangenome

        def __init__(self, ds: DataPortalDataset):
            self.ds = ds
            self.adata = self._make_adata()

            # Get the pangenome that was used for this analysis
            pg_dataset_id = (
                self.ds
                .params
                .additional_properties
                ['inputs']
                ['pangenome']
                .rsplit("/", 1)[-1]
            )
            try:
                pg_dataset = next((
                    pangenome_dataset
                    for pangenome_dataset in pangenome_datasets
                    if pangenome_dataset.id == pg_dataset_id
                ))
            except StopIteration:
                raise Exception(f"The reference pangenome ({pg_dataset_id}) was not found")

            self.pg = Pangenome(
                pg_dataset,
                min_prop=float(query_params.get("min_prop", 0.5))
            )

            self.calculate_relative_sequencing_depth()

            # Get the dataset containing the WGS reads used for this analysis
            self.ngs = self.ds.source_datasets[0]

        def to_df(self, layer: str, rename_bins=False, rename_specimens=False):
            # DataFrame of relative abundance annotated by the source datasets
            df = (
                self.adata
                .to_df(layer=layer)
                .drop(
                    columns=[np.nan],
                    index=[np.nan],
                    errors="ignore"
                )
            )

            if rename_bins:
                df = df.rename(columns=self.rename_bin)
            if rename_specimens:
                df = df.rename(index=self.rename_specimen)
            return df

        def obs(self, rename=False):
            obs = self.adata.obs
            obs.drop(np.nan, errors="ignore", inplace=True)
            if rename:
                obs = obs.rename(index=self.rename_specimen)
            return obs

        def var(self, rename=False):
            var = self.adata.var
            var.drop(np.nan, errors="ignore", inplace=True)
            if rename:
                var = var.rename(index=self.rename_bin)
            return var

        def rename_bin(self, bin_name):
            return f"{bin_name} {self.pg.ds.name} ({self.pg.ds.id.split('-')[0]})"

        def rename_specimen(self, specimen_name):
            return f"{specimen_name} {self.ngs.name} ({self.ngs.id.split('-')[0]})"

        def list_genes(self, rename=False):
            return {
                (
                    self.rename_bin(bin_id)
                    if rename
                    else bin_id
                ): bin_contents["combined_name"].tolist()
                for bin_id, bin_contents in self.pg.bin_contents.items()
            }

        def calculate_relative_sequencing_depth(self):

            # Annotation of all genes - including 'bin' and 'length'
            gene_annot = self.read_csv("data/csv/metagenome.genes.var.csv.gz")
            # Calculate the aggregate gene length per bin
            self.adata.var = self.adata.var.assign(
                combined_length_aa=gene_annot.groupby("bin")["length"].sum()
            )

            # Sequencing Depth in units of RPK (reads per kilobase)
            # (n_reads / (gene_length_aa * 3 / 1000))
            self.adata.layers["rpk"] = (
                self.adata.to_df()
                /
                (self.adata.var["combined_length_aa"] * 3. / 1000.)
            )

            # Adjusted for sequencing depth - RPKM (reads per kilobase per million reads)
            # Calculated both with total reads, and with aligned reads
            self.adata.layers["rpkm_total"] = (
                self.adata.to_df(layer="rpk").T / (self.adata.obs["genes:tot_reads"] / 1e6)
            ).T
            self.adata.layers["rpkm_aligned"] = (
                self.adata.to_df(layer="rpk").T
                / (self.adata.obs["genes:aligned_reads"] / 1e6)
            ).T

        def _make_adata(self):
            # Observation (sample) metadata
            obs = self.read_csv("data/csv/metagenome.obs.csv.gz", index_col=0)
            # Gene Bin metadata (just n_genes)
            var = self.read_csv("data/csv/metagenome.bins.var.csv.gz", index_col=0)
            # Number of reads per bin, per observation
            X = self.read_csv("data/csv/metagenome.bins.X.csv.gz", index_col=0).map(int)

            return AnnData(
                X=X,
                obs=obs.reindex(X.index).fillna("None"),
                var=var.reindex(index=X.columns)
            )

        def read_csv(self, fp: str, **kwargs):
            return read_csv_cached(
                self.ds.project_id,
                self.ds.id,
                fp,
                **kwargs
            )


    # Cache the creation of pangenome objects
    def make_metagenome(ds: DataPortalDataset):
        with mo.status.spinner("Loading data..."):
            return Metagenome(ds)
    return Metagenome, make_metagenome


@app.cell(hide_code=True)
def _(np):
    def format_log_ticks(min_val: float, max_val: float):

        # Get the points to make ticks for in the color scale
        abund_ticks_log = np.arange(np.ceil(np.log10(min_val)), np.log10(max_val), step=1)

        # Format the text label for each of those ticks
        ticktext = [
            (
                f"{10**val:,.0f}"
                if val >= 0
                else f"{10**val}"
            )
            for val in abund_ticks_log
        ]

        return abund_ticks_log, ticktext
    return (format_log_ticks,)


@app.cell
def _(mo):
    get_cluster_args_cat1, set_cluster_args_cat1 = mo.state("specimen_clusters")
    get_cluster_args_cat2, set_cluster_args_cat2 = mo.state(None)
    return (
        get_cluster_args_cat1,
        get_cluster_args_cat2,
        set_cluster_args_cat1,
        set_cluster_args_cat2,
    )


@app.cell(hide_code=True)
def define_inspect_metagenome(
    AnnData,
    Dict,
    List,
    Metagenome,
    Optional,
    Patch,
    Set,
    copy,
    defaultdict,
    format_log_ticks,
    get_cluster_args_cat1,
    get_cluster_args_cat2,
    get_ima_annot_specimens_by,
    get_ima_height,
    get_ima_label_specimens_by,
    get_ima_show_bin_labels,
    get_ima_show_specimen_labels,
    get_ima_width,
    groupby,
    lru_cache,
    mo,
    np,
    pd,
    plt,
    px,
    set_cluster_args_cat1,
    set_cluster_args_cat2,
    set_ima_annot_specimens_by,
    set_ima_height,
    set_ima_label_specimens_by,
    set_ima_show_bin_labels,
    set_ima_show_specimen_labels,
    set_ima_width,
    sklearn,
    sns,
    sort_axis,
    stats,
):
    from hashlib import sha256
    from pandas.util import hash_pandas_object


    class HashableDataFrame(pd.DataFrame):
        def __init__(self, obj):
            super().__init__(obj)

        def __hash__(self):
            hash_value = sha256(hash_pandas_object(self, index=True).values)
            hash_value = hash(hash_value.hexdigest())
            return hash_value

        def __eq__(self, other):
            return self.equals(other)


    class InspectMetagenome:
        mgs: List[Metagenome]
        datasets_df: pd.DataFrame
        _rename_specimens: bool
        _rename_bins: bool    

        # Key: Bin ID - Value: List of gene annotations
        genes_in_bins: Dict[str, List[str]]
        # Key: Unique combinations of contiguous words in gene annotations - Value: List of Bin IDs containing it
        annot_phrases: Dict[str, Set[str]]

        def __init__(self, mgs: List[Metagenome]):
            self.mgs = mgs

            # Key the metagenomes by the same pangenome, or the same ngs
            self._mgs_by_pg = defaultdict(list)
            self._mgs_by_ngs = defaultdict(list)
            for mg in self.mgs:
                self._mgs_by_pg[mg.pg.ds.id].append(mg)
                self._mgs_by_ngs[mg.ngs.id].append(mg)

            # Make a table of all of the pangenomes that each metagenome uses
            self.datasets_df = pd.DataFrame([
                dict(
                    ngs_id=mg.ngs.id,
                    pangenome_id=mg.pg.ds.id,
                    pangenome_name=mg.pg.ds.name,
                    mg_id=mg.ds.id
                )
                for mg in self.mgs
            ])

            # Only rename the specimens if there are multiple NGS datasets
            self._rename_specimens = self.datasets_df["ngs_id"].nunique() > 1

            # Only rename the bins if there are multiple pangenomes
            self._rename_bins = self.datasets_df["pangenome_id"].nunique() > 1

            # Make a combined relative abundance table
            # Group each distinct ngs dataset analyzed with different pangenomes
            # Concatenate each distinct ngs dataset
            rpkm_aligned = self.make_df(layer="rpkm_aligned")
            rpkm_total = self.make_df(layer="rpkm_total")

            # Make the combined metadata table, keeping in mind that the obs information
            # will only be populated when a sample has genes detected for a particular metagenome
            obs = self.make_obs()

            var = pd.concat(
                [
                    same_pg_mgs[0].var(rename=self._rename_bins)
                    for pg_id, same_pg_mgs in self._mgs_by_pg.items()
                ],
                axis=0
            )

            self.adata = AnnData(
                rpkm_aligned,
                obs=obs.reindex(rpkm_aligned.index),
                var=var.reindex(rpkm_aligned.columns),
                layers=dict(rpkm_total=rpkm_total)
            )

            # Save off the gene names for all bins
            self.genes_in_bins = {
                bin_id: gene_list
                for same_pg_mgs in self._mgs_by_pg.values()
                for bin_id, gene_list in same_pg_mgs[0].list_genes(rename=self._rename_bins).items()
            }

            # Tally up all of the unique annotation phrases in the pangenome, and which bins they are found in
            self.find_annot_phrases()

        def find_annot_phrases(self):
            """Find every unique combination of contiguous words, and make a list of which bins they are found within."""
            self.annot_phrases = defaultdict(set)

            for bin_id, gene_list in self.genes_in_bins.items():
                for gene_annot in gene_list:
                    words = gene_annot.rstrip("\n").split(" ")
                    for i in range(len(words)):
                        for j in range(len(words)):
                            if i < j:
                                self.annot_phrases[' '.join(words[i:j])].add(bin_id)

            # Drop any phrases which are found in the exact same set of bins as a longer phrase that contains it
            to_drop = set([
                phrase1
                for phrase1, bins1 in self.annot_phrases.items()
                for phrase2, bins2 in self.annot_phrases.items()
                if phrase1 != phrase2 and bins1 == bins2 and phrase1 in phrase2
            ])
            self.annot_phrases = {
                kw: val
                for kw, val in self.annot_phrases.items()
                if kw not in to_drop
            }

        def make_obs(self) -> pd.DataFrame:
            return pd.concat(
                [
                    self.make_obs_per_ngs(list(same_ngs_mgs))
                    for ngs_id, same_ngs_mgs in groupby(
                        self.mgs,
                        lambda mg: mg.ngs.id
                    )                    
                ],
                axis=0
            )

        def make_obs_per_ngs(self, same_ngs_mgs: List[Metagenome]) -> pd.DataFrame:
            obs = same_ngs_mgs[0].obs(rename=self._rename_specimens)

            if len(same_ngs_mgs) > 1:

                for ngs in same_ngs_mgs[1:]:
                    next_obs = ngs.obs(rename=self._rename_specimens)

                    if set(next_obs.index.values) > set(obs.index.values):
                        obs = pd.concat([
                            obs,
                            next_obs.reindex(index=set(next_obs.index.values) - set(obs.index.values))
                        ])

            # Annotate the NGS dataset
            ngs = same_ngs_mgs[0].ngs
            obs = obs.assign(
                ngs_dataset_id=ngs.id,
                ngs_dataset_name=ngs.name,
            )
            return obs

        def make_df(self, layer: str):
            """Get the abundances from all metagenomes, using a particular layer of data."""
            return pd.concat(
                [
                    pd.concat(
                        [
                            mg.to_df(
                                layer=layer,
                                rename_specimens=self._rename_specimens,
                                rename_bins=self._rename_bins
                            )
                            for mg in same_ngs_mgs
                        ],
                        axis=1
                    )
                    for ngs_id, same_ngs_mgs in groupby(
                        self.mgs,
                        lambda mg: mg.ngs.id
                    )
                ],
                axis=0
            ).fillna(0)

        def header(self):
            """Header text."""

            return mo.md(f"""
    ### Inspect Metagenome: Analysis Options

    ({self.adata.shape[0]:,} specimens x {self.adata.shape[1]:,} bins)
    """)

        def analysis_args(self):
            """Top-level args."""

            return mo.md("""
     - {per_total_or_aligned}
     - {include_bins}
     - {filter_specimens_query}
     - {n_clusters}
            """).batch(
                per_total_or_aligned=mo.ui.dropdown(
                    label="Calculate Sequencing Depth Relative To:",
                    options=["All Reads", "Pangenome-Aligned Reads"] + list(self.adata.var_names),
                    value="All Reads"
                ),
                include_bins=mo.ui.multiselect(
                    label="Use Specific Bins (default: all):",
                    value=[],
                    options=self.adata.var_names
                ),
                filter_specimens_query=mo.ui.text(
                    label="Filter Specimens by Query Expression (default: all)",
                    placeholder="colName == 'Group A'"
                ),
                n_clusters=mo.ui.number(
                    label="K-Means Clustering - K:",
                    value=5,
                    step=1,
                    start=2
                )
            )


    class InspectMetagenomeAnalysis:
        adata: AnnData
        genes_in_bins: Dict[str, List[str]]
        annot_phrases: Dict[str, Set[str]]
        abund: pd.DataFrame
        log_abund: pd.DataFrame
        lower_bound: float
        per_total_or_aligned: str

        def __init__(
            self,
            adata: AnnData,
            genes_in_bins: Dict[str, List[str]],
            annot_phrases: Dict[str, Set[str]],
            per_total_or_aligned: str,
            include_bins: List[str],
            filter_specimens_query: Optional[str],
            n_clusters: int,
        ):
            # Attach the AnnData object (generated by the InspectMetagenome object defined above)
            self.adata = adata

            # Attach the bin annotations
            self.genes_in_bins = genes_in_bins
            self.annot_phrases = annot_phrases

            # Get the abundances requested by the user
            self.per_total_or_aligned = per_total_or_aligned
            if self.per_total_or_aligned == "Pangenome-Aligned Reads":
                _abund = self.adata.to_df()
            elif self.per_total_or_aligned == "All Reads":
                _abund = self.adata.to_df(layer="rpkm_total")
            else:
                assert self.per_total_or_aligned in self.adata.var_names
                _abund = self.adata.to_df(layer="rpkm_total")
                _abund = (
                    _abund
                    .div(_abund[self.per_total_or_aligned], axis=0)
                    .drop(columns=[self.per_total_or_aligned])
                )

            # Optionally filter bins
            if include_bins is not None and len(include_bins) > 0:
                _bins_to_plot = include_bins
            else:
                _bins_to_plot = _abund.columns.values

            # If the user specified a query string for the specimens
            if filter_specimens_query is not None and len(filter_specimens_query) > 0:
                try:
                    filtered_specimens = self.adata.obs.query(filter_specimens_query).index
                except ValueError as e:
                    return mo.md(f"Could not evaluate query: {str(e)}")
                if len(filtered_specimens) == 0:
                    return mo.md("No specimens match the provided query")
                if len(filtered_specimens) < 2:
                    return mo.md("Only a single specimen matches the provided query")
                _abund = _abund.reindex(index=filtered_specimens)

            self.abund = _abund.reindex(columns=_bins_to_plot)

            if len(_bins_to_plot) < 2:
                return mo.md("""Please select multiple bins to plot""")

            # Calculate the log abundance for the color scale
            self.lower_bound = self.abund.apply(lambda c: c[c > 0].min()).min()
            self.log_abund = (
                self.abund
                .clip(
                    lower=self.lower_bound
                )
                .apply(np.log10)
            )

            self.specimen_order = sort_axis(self.log_abund, metric="euclidean", method="average")
            self.bin_order = sort_axis(self.log_abund.T, metric="euclidean", method="average")

            self.abund = self.abund.reindex(index=self.specimen_order, columns=self.bin_order)
            self.log_abund = self.log_abund.reindex(index=self.specimen_order, columns=self.bin_order)

            # Perform k-means clustering
            # Rank the bins based on how different they are between those clusters using the kruskal wallis H test
            self.adata.obs["specimen_clusters"], kw = run_kmeans_clustering(self.log_abund, n_clusters)
            self.adata.varm["specimen_clusters_kw"] = kw.reindex(index=self.adata.var.index)

        @property
        def bin_names(self):
            return list(self.adata.var_names)

        @property
        def obs_cnames(self):
            return list(self.adata.obs.columns.values)

        @property
        def obs_cnames_and_bin_names(self):
            return self.obs_cnames + self.bin_names

        def heatmap_args(self):
            """Top-level args."""

            return mo.md("""
    ### Inspect Metagenome: Heatmap Options

     - Show Specimens Labels {show_specimen_labels}
     - Custom Specimen Label: {label_specimens_by}
     - Annotate Specimens By: {annot_specimens_by}
     - Show Pangenome Bin Labels {show_bin_labels}
     - {height}
     - {width}
            """).batch(
                show_specimen_labels=mo.ui.checkbox(
                    value=get_ima_show_specimen_labels(),
                    on_change=set_ima_show_specimen_labels
                ),
                label_specimens_by=mo.ui.dropdown(
                    options=self.obs_cnames,
                    value=get_ima_label_specimens_by(),
                    on_change=set_ima_label_specimens_by,
                    searchable=True
                ),
                annot_specimens_by=mo.ui.multiselect(
                    options=self.obs_cnames,
                    value=get_ima_annot_specimens_by(),
                    on_change=set_ima_annot_specimens_by
                ),
                show_bin_labels=mo.ui.checkbox(
                    value=get_ima_show_bin_labels(),
                    on_change=set_ima_show_bin_labels
                ),
                height=mo.ui.number(
                    label="Figure Height",
                    start=1,
                    value=get_ima_height(),
                    on_change=set_ima_height
                ),
                width=mo.ui.number(
                    label="Figure Width",
                    start=1,
                    value=get_ima_width(),
                    on_change=set_ima_width
                )
            )

        def heatmap_plot(
            self,
            show_specimen_labels: bool,
            label_specimens_by: str,
            annot_specimens_by: List[str],
            show_bin_labels: bool,
            width: int,
            height: int
        ):

            # Get the points to make ticks for in the color scale
            abund_ticks_log, ticktext = format_log_ticks(self.lower_bound, self.abund.max().max())

            # Set up the row annotations
            if len(annot_specimens_by) > 0:
                row_colors, row_cmap = make_df_cmap(
                    self.adata
                    .obs
                    .reindex(columns=annot_specimens_by)
                    .reindex(self.specimen_order)
                )
            else:
                row_colors = None
                row_cmap = None

            fig = sns.clustermap(
                self.log_abund,
                cmap="Blues",
                yticklabels=(
                    (
                        "auto"
                        if label_specimens_by is None
                        else self.adata.obs.reindex(self.specimen_order)[label_specimens_by].values
                    )
                    if show_specimen_labels
                    else False
                ),
                xticklabels="auto" if show_bin_labels else False,
                figsize=(width, height),
                cbar_pos=(0, 0.5, .05, .3),
                cbar_kws=dict(
                    ticks=abund_ticks_log
                ),
                row_cluster=False,
                col_cluster=False,
                dendrogram_ratio=(0.15, 0.01),
                row_colors=row_colors
            )
            fig.fig.suptitle(f"Relative Sequencing Depth - {self.per_total_or_aligned}", y=1.05)
            fig.ax_cbar.set_title("RPKM")
            fig.ax_cbar.set_yticklabels(ticktext)
            fig.ax_heatmap.set_ylabel(None)

            if row_cmap is not None:
                legends = [
                    fig.ax_row_dendrogram.legend(
                        [Patch(facecolor=_cmap[name]) for name in _cmap],
                        _cmap,
                        title=_label,
                        bbox_to_anchor=(
                            1,
                            1 - (_ix / (len(row_cmap) + 1))
                        ),
                        bbox_transform=plt.gcf().transFigure,
                        loc='upper left'
                    )
                    for _ix, (_label, _cmap) in enumerate(row_cmap.items())
                ]
                if len(legends) > 1:
                    for legend in legends[:-1]:
                        fig.ax_row_dendrogram.add_artist(legend)

            return plt.gca()

        def scatter_args(self):
            """Top-level args."""

            return mo.md("""
    ### Inspect Metagenome: Scatter Options

     - Custom Specimen Label: {label_specimens_by}
     - Color Specimens By: {color_specimens_by}
     - Hover Data (optional): {hover_data}
     - {n_dims}
     - {perplexity}
     - {height}
     - {width}
            """).batch(
                label_specimens_by=mo.ui.dropdown(
                    options=self.obs_cnames,
                    searchable=True
                ),
                color_specimens_by=mo.ui.dropdown(
                    options=self.obs_cnames_and_bin_names,
                    searchable=True
                ),
                hover_data=mo.ui.multiselect(
                    options=self.obs_cnames,
                    value=[]
                ),
                n_dims=mo.ui.radio(
                    label="t-SNE - Dimensions:",
                    options=["2D", "3D"],
                    value="2D"
                ),
                perplexity=mo.ui.number(
                    label="t-SNE - perplexity:",
                    start=1.,
                    value=float(min(30, self.adata.n_obs-1)),
                    stop=self.adata.n_obs
                ),
                height=mo.ui.number(
                    label="Figure Height",
                    start=1,
                    value=500
                ),
                width=mo.ui.number(
                    label="Figure Width",
                    start=1,
                    value=500
                )
            )

        def scatter_plot(
            self,
            label_specimens_by: str,
            color_specimens_by: str,
            hover_data: list,
            n_dims: str,
            perplexity: float,
            width: int,
            height: int
        ):

            # Use the log abundances to perform the t-SNE embedding
            tsne_coords = copy(run_tsne(
                HashableDataFrame(self.log_abund),
                n_components=2 if n_dims == "2D" else 3,
                perplexity=perplexity,
                random_state=0
            ))

            # Optionally assign a color
            if color_specimens_by is not None:
                tsne_coords = tsne_coords.assign(**{
                    color_specimens_by: (
                        self.adata.obs[color_specimens_by]
                        if color_specimens_by in self.adata.obs.columns
                        else
                        self.log_abund[color_specimens_by]
                    )
                })

            # Add any hover data that was specified
            for cname in hover_data:
                if cname not in tsne_coords.columns.values:
                    tsne_coords = tsne_coords.assign(**{
                        cname: self.adata.obs[cname]
                    })

            # Optionally assign a label
            if label_specimens_by is not None and label_specimens_by not in tsne_coords.columns.values:
                tsne_coords = tsne_coords.assign(**{
                    label_specimens_by: self.adata.obs[label_specimens_by]
                })
            else:
                label_specimens_by = tsne_coords.index.name
                tsne_coords.reset_index(inplace=True)

            # Things that are the same for both plot types
            common_kwargs = dict(
                data_frame=tsne_coords,
                x="t-SNE 1",
                y="t-SNE 2",
                hover_name=label_specimens_by,
                hover_data=hover_data,
                template="simple_white",
                width=width,
                height=height
            )
            if color_specimens_by is not None:
                common_kwargs["color"] = color_specimens_by

            # Switch based on the 2D or 3D scatter option
            if n_dims == "2D":
                fig = px.scatter(
                    **common_kwargs
                )
            else:
                fig = px.scatter_3d(
                    z="t-SNE 3",
                    **common_kwargs
                )
            return fig

        def cluster_args(self):
            """User input for comparing the distribution of specimen annotations (e.g. clusters vs. anything)."""
            return mo.md("""
    ### Inspect Metagenomes: Compare Specimen Annotations

    Plot the distribution of specimens across different annotation groups, for example
    comparing treatment vs. control specimens between the clusters assigned by unsupervised
    k-means based on organism abundances.

     - {cat1}
     - {cat2}
        """).batch(
                cat1=mo.ui.dropdown(
                    label="Annotation 1:",
                    options=self.obs_cnames,
                    value=(
                        get_cluster_args_cat1()
                        if get_cluster_args_cat1() in self.obs_cnames
                        else self.obs_cnames[-1]
                    ),
                    on_change=set_cluster_args_cat1,
                    searchable=True
                ),
                cat2=mo.ui.dropdown(
                    label="Annotation 2:",
                    options=self.obs_cnames,
                    value=(
                        get_cluster_args_cat2()
                        if get_cluster_args_cat2() in self.obs_cnames
                        else None
                    ),
                    on_change=set_cluster_args_cat2,
                    searchable=True
                )
            )

        def cluster_secondary_args(self, cat1: str, cat2: str):
            cat1_options = self.adata.obs[cat1].dropna().unique() if cat1 is not None else []
            cat2_options = self.adata.obs[cat2].dropna().unique() if cat2 is not None else []

            return mo.md("""
     - {cat1_groups}
     - {cat2_groups}
     - {norm}
     - {barmode}
     - {height}
     - {width}
        """).batch(
                cat1_groups=mo.ui.multiselect(
                    label=f"Include Groups from {cat1}",
                    options=cat1_options,
                    value=cat1_options
                ),
                cat2_groups=mo.ui.multiselect(
                    label=f"Include Groups from {cat2}",
                    options=cat2_options,
                    value=cat2_options
                ),
                norm=mo.ui.dropdown(
                    label="Display Values",
                    options=[
                        "Number of Specimens",
                        f"Percent of Specimens per {cat1} Group",
                        f"Percent of Specimens per {cat2} Group"
                    ],
                    value="Number of Specimens"
                ),
                barmode=mo.ui.radio(
                    label="Bar Mode:",
                    options=["stack", "group"],
                    value="group"
                ),
                height=mo.ui.number(
                    label="Figure Height",
                    start=1,
                    value=500
                ),
                width=mo.ui.number(
                    label="Figure Width",
                    start=1,
                    value=500
                )
            )

        def cluster_plot(
            self,
            cat1: str,
            cat1_groups: List[str],
            cat2: str,
            cat2_groups: List[str],
            norm: str,
            barmode: str,
            height: int,
            width: int
        ):
            if cat2 is None:
                return mo.md("Please select a category to compare.")
            if cat2 == cat1:
                return mo.md("Please select two different categories to compare.")
            if len(cat1_groups) < 2:
                return mo.md(f"Please select more than 1 group from {cat1}")
            if len(cat2_groups) < 2:
                return mo.md(f"Please select more than 1 group from {cat2}")

            # Get the contingency table
            ct = pd.crosstab(
                self.adata.obs[cat1],
                self.adata.obs[cat2]
            ).reindex(
                columns=cat2_groups,
                index=cat1_groups
            )

            chi2_res = stats.chi2_contingency(ct.values)
            pvalue = (
                (
                    f"{chi2_res.pvalue:.2E}"
                    if chi2_res.pvalue < 0.001 else
                    f"{chi2_res.pvalue:.3}"
                )
                if chi2_res.pvalue < 0.01
                else f"{chi2_res.pvalue:.2}"
            )

            # Get the values to plot
            if norm == f"Percent of Specimens per {cat1} Group":
                ct = ct.apply(lambda r: r / r.sum(), axis=1)
            elif norm == f"Percent of Specimens per {cat2} Group":
                ct = ct.apply(lambda r: r / r.sum(), axis=0)

            plot_df = (
                ct
                .reset_index()
                .melt(id_vars=cat1)
            )

            fig = px.bar(
                data_frame=plot_df,
                x=cat1,
                y="value",
                color=cat2,
                template="simple_white",
                barmode=barmode,
                title=f"Chi2 p-value: {pvalue}",
                width=width,
                height=height
            )
            return fig

        def cluster_dist_args(self):
            """User input for comparing bin abundances across any specimen annotation (e.g. clusters)."""

            return mo.md("""
    ### Inspect Metagenomes: Compare Bin Abundances

    Plot the relative abundance of any bin, comparing across any grouping of specimens (e.g. k-means clusters).

    - {kind}
    - {x}
    - {y}
    - {hue}
    - {height}
    - {aspect}
            """).batch(
                kind=mo.ui.dropdown(
                    label="Plot Type:",
                    value="boxen",
                    options=[
                        "boxen",
                        "box",
                        "violin",
                        "strip",
                        "swarm",
                        "point",
                        "bar"
                    ]
                ),
                x=mo.ui.dropdown(
                    label="X-axis:",
                    options=self.obs_cnames,
                    value="specimen_clusters",
                    searchable=True
                ),
                y=mo.ui.dropdown(
                    label="Y-axis:",
                    options=self.bin_names,
                    value=(
                        self.adata
                        .varm["specimen_clusters_kw"]
                        .dropna()
                        .sort_values(by="pvalue")
                        .index.values[0]
                    ),
                    searchable=True
                ),
                hue=mo.ui.dropdown(
                    label="Hue:",
                    options=self.obs_cnames,
                    searchable=True
                ),
                height=mo.ui.number(
                    label="Height:",
                    value=5.,
                    start=1.
                ),
                aspect=mo.ui.number(
                    label="Aspect Ratio:",
                    value=1.,
                    start=0.1,
                )
            )

        def cluster_dist_plot(
            self,
            kind: str,
            x: str,
            y: str,
            hue: str,
            height: float,
            aspect: float,
        ):
            plot_df = pd.concat([self.adata.obs, self.log_abund], axis=1)

            try:
                fig = sns.catplot(
                    plot_df,
                    kind=kind,
                    x=x,
                    y=y,
                    hue=hue,
                    height=height,
                    aspect=aspect
                )
            except Exception as e:
                return str(e)

            return plt.gca()


    @lru_cache
    def run_tsne(df: HashableDataFrame, n_components: int, perplexity: float, random_state):
        tsne = sklearn.manifold.TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state
        )
        return pd.DataFrame(
            tsne.fit_transform(df.values),
            index=df.index,
            columns = [
                f"t-SNE {ix+1}"
                for ix in range(n_components)
            ]
        )


    def run_kmeans_clustering(df: pd.DataFrame, n_clusters: int) -> pd.Series:
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(df.values)

        # Use the Kruskal Wallis H test to rank the features and how much they vary between groups
        kw = pd.DataFrame([
            dict(zip(
                ["statistic", "pvalue"],
                try_kruskal(
                    *[
                        value_group
                        for _, value_group in col.groupby(labels)
                    ]
                )
            ))
            for _, col in df.items()
        ], index=df.columns)
        return pd.Series(labels, index=df.index).apply(str), kw


    def try_kruskal(*samples):
        try:
            return stats.kruskal(*samples)
        except ValueError:
            return (None, None)


    def make_df_cmap(df: pd.DataFrame, palette="tab10", **kwargs):
        """Make a colormap for each column in a DataFrame."""

        cmap = dict()
        colors_df = dict()

        for cname, cvals in df.items():
            unique_cvals = cvals.drop_duplicates().sort_values().values
            cmap[cname] = dict(zip(
                unique_cvals,
                sns.color_palette(palette, len(unique_cvals), **kwargs)
            ))
            colors_df[cname] = cvals.apply(cmap[cname].get)

        return pd.DataFrame(colors_df), cmap
    return InspectMetagenome, InspectMetagenomeAnalysis, make_df_cmap


@app.cell
def _(mo):
    get_ima_show_specimen_labels, set_ima_show_specimen_labels = mo.state(False)
    get_ima_label_specimens_by, set_ima_label_specimens_by = mo.state(None)
    get_ima_annot_specimens_by, set_ima_annot_specimens_by = mo.state([])
    get_ima_show_bin_labels, set_ima_show_bin_labels = mo.state(True)
    get_ima_height, set_ima_height = mo.state(8)
    get_ima_width, set_ima_width = mo.state(8)
    return (
        get_ima_annot_specimens_by,
        get_ima_height,
        get_ima_label_specimens_by,
        get_ima_show_bin_labels,
        get_ima_show_specimen_labels,
        get_ima_width,
        set_ima_annot_specimens_by,
        set_ima_height,
        set_ima_label_specimens_by,
        set_ima_show_bin_labels,
        set_ima_show_specimen_labels,
        set_ima_width,
    )


@app.cell
def make_inspect_metagenome(
    InspectMetagenome,
    inspect_metagenome_datasets,
    make_metagenome,
    mo,
):
    with mo.status.spinner("Building Metagenome"):
        inspect_metagenome = InspectMetagenome([
            make_metagenome(metagenome_dataset)
            for metagenome_dataset in inspect_metagenome_datasets
        ])
    return (inspect_metagenome,)


@app.cell
def _(inspect_metagenome):
    inspect_metagenome.header()
    return


@app.cell
def _(inspect_metagenome):
    inspect_metagenome_args = inspect_metagenome.analysis_args()
    inspect_metagenome_args
    return (inspect_metagenome_args,)


@app.cell
def _(InspectMetagenomeAnalysis, inspect_metagenome, inspect_metagenome_args):
    inspect_metagenome_analysis = InspectMetagenomeAnalysis(
        adata=inspect_metagenome.adata,
        genes_in_bins=inspect_metagenome.genes_in_bins,
        annot_phrases=inspect_metagenome.annot_phrases,
        **inspect_metagenome_args.value
    )
    return (inspect_metagenome_analysis,)


@app.cell
def _(inspect_metagenome_analysis):
    inspect_metagenome_heatmap_args = inspect_metagenome_analysis.heatmap_args()
    inspect_metagenome_heatmap_args
    return (inspect_metagenome_heatmap_args,)


@app.cell
def _(inspect_metagenome_analysis, inspect_metagenome_heatmap_args):
    inspect_metagenome_analysis.heatmap_plot(**inspect_metagenome_heatmap_args.value)
    return


@app.cell
def _(inspect_metagenome_analysis):
    inspect_metagenome_cluster_args = inspect_metagenome_analysis.cluster_args()
    inspect_metagenome_cluster_args
    return (inspect_metagenome_cluster_args,)


@app.cell
def _(inspect_metagenome_cluster_args, mo):
    (
        mo.md("Please select two categories to compare")
        if inspect_metagenome_cluster_args.value["cat1"] is None or inspect_metagenome_cluster_args.value["cat2"] is None
        else None
    )
    return


@app.cell
def _(inspect_metagenome_analysis, inspect_metagenome_cluster_args, mo):
    mo.stop(inspect_metagenome_cluster_args.value["cat1"] is None or inspect_metagenome_cluster_args.value["cat2"] is None)
    inspect_metagenome_cluster_secondary_args = inspect_metagenome_analysis.cluster_secondary_args(**inspect_metagenome_cluster_args.value)
    inspect_metagenome_cluster_secondary_args
    return (inspect_metagenome_cluster_secondary_args,)


@app.cell
def _(
    inspect_metagenome_analysis,
    inspect_metagenome_cluster_args,
    inspect_metagenome_cluster_secondary_args,
):
    inspect_metagenome_analysis.cluster_plot(
        **inspect_metagenome_cluster_args.value,
        **inspect_metagenome_cluster_secondary_args.value
    )
    return


@app.cell
def _(inspect_metagenome_analysis):
    inspect_metagenome_scatter_args = inspect_metagenome_analysis.scatter_args()
    inspect_metagenome_scatter_args
    return (inspect_metagenome_scatter_args,)


@app.cell
def _(inspect_metagenome_analysis, inspect_metagenome_scatter_args):
    inspect_metagenome_analysis.scatter_plot(**inspect_metagenome_scatter_args.value)
    return


@app.cell
def _(inspect_metagenome_analysis):
    inspect_metagenome_cluster_dist_args = inspect_metagenome_analysis.cluster_dist_args()
    inspect_metagenome_cluster_dist_args
    return (inspect_metagenome_cluster_dist_args,)


@app.cell
def _(inspect_metagenome_analysis, inspect_metagenome_cluster_dist_args):
    inspect_metagenome_analysis.cluster_dist_plot(**inspect_metagenome_cluster_dist_args.value)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Compare Metagenomes

    Perform statistical analysis comparing the relative abundance of different pangenome elements between groups of metagenome specimens.
    """
    )
    return


@app.cell
def class_comparemetagenometool(
    AnnData,
    Dict,
    List,
    Patch,
    Set,
    format_log_ticks,
    inspect_metagenome_analysis,
    lru_cache,
    make_df_cmap,
    mo,
    pd,
    plt,
    px,
    sns,
    sort_axis,
    stats,
):
    class CompareMetagenomeTool:
        """Base class used for all analysis methods."""
        name: str
        description: str

        _ready_to_plot: bool

        log_abund: pd.DataFrame
        abund: pd.DataFrame
        adata: AnnData

        res: pd.DataFrame

        def __init__(self):
            self.adata = inspect_metagenome_analysis.adata
            self.log_abund = inspect_metagenome_analysis.log_abund
            self.abund = inspect_metagenome_analysis.abund
            self._ready_to_plot = False
            self.obs_cnames = inspect_metagenome_analysis.obs_cnames
            self.lower_bound = inspect_metagenome_analysis.lower_bound
            self.per_total_or_aligned = inspect_metagenome_analysis.per_total_or_aligned
            self.genes_in_bins = inspect_metagenome_analysis.genes_in_bins
            self.annot_phrases = inspect_metagenome_analysis.annot_phrases

        def primary_args(self):
            return mo.md("").batch()

        def secondary_args(self, **kwargs):
            return mo.md("").batch()

        def run_analysis(self, **kwargs):
            pass

        def primary_plot_args(self, **kwargs):
            return mo.md("").batch()

        def make_primary_plot(self, **kwargs):
            return self.res

        def secondary_plot_args(self,grouping_cname: str, **kwargs):
            if self.res is None:
                return

            return mo.md("""
    ### Compare Metagenomes: Heatmap Options

     - Show Bins in Heatmap: {heatmap_show_bins}
     - Show Specimens Labels {heatmap_show_specimen_labels}
     - Custom Specimen Label: {heatmap_label_specimens_by}
     - Annotate Specimens By: {heatmap_annot_specimens_by}
     - Show Pangenome Bin Labels {heatmap_show_bin_labels}
     - Figure Height: {heatmap_height}
     - Figure Width: {heatmap_width}
            """).batch(
                heatmap_show_bins=mo.ui.multiselect(
                    options=self.res['bin'].values,
                    value=self.res['bin'].head(20).values
                ),
                heatmap_show_specimen_labels=mo.ui.checkbox(
                    value=False
                ),
                heatmap_label_specimens_by=mo.ui.dropdown(
                    options=self.obs_cnames,
                    searchable=True
                ),
                heatmap_annot_specimens_by=mo.ui.multiselect(
                    options=self.obs_cnames,
                    value=[grouping_cname]
                ),
                heatmap_show_bin_labels=mo.ui.checkbox(
                    value=True
                ),
                heatmap_height=mo.ui.number(
                    start=1,
                    value=8
                ),
                heatmap_width=mo.ui.number(
                    start=1,
                    value=8
                )
            )

        def make_secondary_plot(
            self,
            heatmap_show_bins: List[str],
            heatmap_show_specimen_labels: bool,
            heatmap_label_specimens_by: str,
            heatmap_annot_specimens_by: List[str],
            heatmap_show_bin_labels: bool,
            heatmap_width: int,
            heatmap_height: int,
            query: str,
            grouping_cname: str,
            include_groups: List[str] = None,
            ref_group: str = None,
            comp_group: str = None,
            **kwargs
        ):
            if include_groups is None:
                include_groups = [ref_group, comp_group]

            # Optionally filter, and get only those groups which were selected
            obs = self._filtered_obs(query)
            obs = obs.loc[obs[grouping_cname].isin(include_groups)]

            # Only use abundances from the selected features
            abund = self.abund.reindex(columns=heatmap_show_bins, index=obs.index)
            log_abund = self.log_abund.reindex(columns=heatmap_show_bins, index=obs.index)

            # Sort the specimens based on those abundances only
            specimen_order = sort_axis(
                log_abund.dropna(how="all", axis=1),
                metric="euclidean",
                method="average"
            )
            bin_order = sort_axis(
                log_abund.dropna(how="all", axis=1).T,
                metric="euclidean",
                method="average"
            )
            obs = obs.reindex(index=specimen_order)
            abund = abund.reindex(index=specimen_order, columns=bin_order)
            log_abund = log_abund.reindex(index=specimen_order, columns=bin_order)

            # Get the points to make ticks for in the color scale
            abund_ticks_log, ticktext = format_log_ticks(
                abund.apply(lambda c: c[c > 0].min()).min(),
                abund.max().max()
            )

            # Set up the row annotations
            if len(heatmap_annot_specimens_by) > 0:
                row_colors, row_cmap = make_df_cmap(
                    obs.reindex(columns=heatmap_annot_specimens_by)
                )
            else:
                row_colors = None
                row_cmap = None

            fig = sns.clustermap(
                log_abund,
                cmap="Blues",
                yticklabels=(
                    (
                        "auto"
                        if heatmap_label_specimens_by is None
                        else obs[heatmap_label_specimens_by].values
                    )
                    if heatmap_show_specimen_labels
                    else False
                ),
                xticklabels="auto" if heatmap_show_bin_labels else False,
                figsize=(heatmap_width, heatmap_height),
                cbar_pos=(0, 0.5, .05, .3),
                cbar_kws=dict(
                    ticks=abund_ticks_log
                ),
                row_cluster=False,
                col_cluster=False,
                dendrogram_ratio=(0.15, 0.01),
                row_colors=row_colors
            )
            fig.fig.suptitle(f"Relative Sequencing Depth - {self.per_total_or_aligned}", y=1.05)
            fig.ax_cbar.set_title("RPKM")
            fig.ax_cbar.set_yticklabels(ticktext)
            fig.ax_heatmap.set_ylabel(None)

            if row_cmap is not None:
                legends = [
                    fig.ax_row_dendrogram.legend(
                        [Patch(facecolor=_cmap[name]) for name in _cmap],
                        _cmap,
                        title=_label,
                        bbox_to_anchor=(
                            1,
                            1 - (_ix / (len(row_cmap) + 1))
                        ),
                        bbox_transform=plt.gcf().transFigure,
                        loc='upper left'
                    )
                    for _ix, (_label, _cmap) in enumerate(row_cmap.items())
                ]
                if len(legends) > 1:
                    for legend in legends[:-1]:
                        fig.ax_row_dendrogram.add_artist(legend)

            return plt.gca()


        def tertiary_plot_args(
            self,
            grouping_cname: str,
            include_groups: List[str] = None,
            ref_group: str = None,
            comp_group: str = None,
            **kwargs
        ):
            if include_groups is None:
                include_groups = [ref_group, comp_group]

            if self.res is None:
                return
            return mo.md("""
     - {select_bins}
     - {single_bin_group_order}
     - {xaxis_label}
     - {plot_type}
     - {single_bin_width}
     - {single_bin_height}
            """).batch(
                select_bins=mo.ui.multiselect(
                    label="Show Bin(s) Across Groups:",
                    options=self.res['bin'].values,
                    value=self.res['bin'].values[:1]
                ),
                single_bin_group_order=mo.ui.multiselect(
                    label="Group Order (deselect and reselect to reorder):",
                    options=include_groups,
                    value=include_groups
                ),
                xaxis_label=mo.ui.text(
                    label="X-Axis Label:",
                    value=grouping_cname
                ),
                plot_type=mo.ui.dropdown(
                    label="Plot Type",
                    options=["Box", "Violin", "Strip"],
                    value="Box"
                ),
                single_bin_height=mo.ui.number(
                    label="Figure Height",
                    start=1,
                    value=400
                ),
                single_bin_width=mo.ui.number(
                    label="Figure Width",
                    start=1,
                    value=600
                )
            )

        def make_tertiary_plot(
            self,
            grouping_cname: str,
            select_bins: str,
            plot_type: str,
            single_bin_group_order: List[str],
            xaxis_label: str,
            single_bin_width: int,
            single_bin_height: int,
            **kwargs
        ):
            plot_df = pd.DataFrame(
                {
                    grouping_cname: (
                        self._groupings.iloc[:, 0]
                        if self._groupings.ndim == 2
                        else self._groupings
                    ),
                    "bin_abund": self.log_abund[select_bins].mean(axis=1)
                }
            ).dropna()

            fig = getattr(px, plot_type.lower())(
                plot_df.loc[
                    plot_df[grouping_cname].isin(single_bin_group_order)
                ],
                x=grouping_cname,
                y="bin_abund",
                category_orders={
                    grouping_cname: single_bin_group_order
                },
                template="simple_white",
                labels={
                    grouping_cname: xaxis_label,
                    "bin_abund": ", ".join(select_bins)
                },
                width=single_bin_width,
                height=single_bin_height
            )
            fig.update_xaxes(type="category")

            return fig

        def _filtered_obs(self, query: str):
            if query is not None and len(query) > 0:
                return self.adata.obs.query(query)
            else:
                return self.adata.obs

        def gene_annot_enrichment_args(self):
            return mo.md("""### Enriched Gene Annotation Phrases

            - {min_n_bins}
            """).batch(
                min_n_bins=mo.ui.number(label="Minimum Number of Bins:", value=2, start=1)
            )

        def gene_annot_enrichment_plot(self, gene_annot_df: pd.DataFrame, min_n_bins: int):
            # For each annotation phrase, test to see if it is enriched in the most strongly associated bins
            return gene_annot_df.query(
                f"statistic < {gene_annot_df['statistic'].median()} and n_bins >= {min_n_bins}"
            )

        def calc_gene_annot_enrichment(self, **kwargs):
            # Get the rank of each bin in this analysis
            bin_rank = dict(zip(self.res['bin'].tolist(), range(self.res.shape[0])))

            # For each phrase, use the MWU to test for enrichment
            return pd.DataFrame([
                self.calc_gene_annot_enrichment_single(phrase, bin_set, bin_rank)
                for phrase, bin_set in self.annot_phrases.items()
            ]).sort_values(by="pvalue")

        def calc_gene_annot_enrichment_single(self, phrase: str, bin_set: Set[str], bin_rank: Dict[str, int]):
            # Get the ranks of all bins that contain this phrase
            bin_ranks = frozenset([bin_rank[bin] for bin in list(bin_set) if bin in bin_rank])
            other_ranks = frozenset(range(len(bin_rank))) - bin_ranks

            # Get the enrichment score and p-value
            statistic, pvalue = self.mwu(bin_ranks, other_ranks)

            return dict(
                phrase=phrase,
                n_words=len(phrase.split(" ")),
                n_bins=len(bin_ranks),
                bins=', '.join(list(bin_set)),
                statistic=statistic,
                pvalue=pvalue
            )

        @lru_cache
        def mwu(_self, bin_ranks, other_ranks):
            if len(bin_ranks) == 0 or len(other_ranks) == 0:
                return None, None
            res = stats.mannwhitneyu(list(bin_ranks), list(other_ranks))
            return res.statistic, res.pvalue

    return (CompareMetagenomeTool,)


@app.cell
def _(mo):
    get_grouping_cname, set_grouping_cname = mo.state("specimen_clusters")
    return get_grouping_cname, set_grouping_cname


@app.cell
def _(
    CompareMetagenomeTool,
    get_grouping_cname,
    mo,
    pd,
    set_grouping_cname,
    stats,
):
    class CompareMetagenomeTwoGroups(CompareMetagenomeTool):
        name = "Compare Two Groups - Test Each Organism"
        description = """
        Test each organism individually for differences in abundance between two specified group.
        Can use either the t-test (parametric) or Mann-Whitney U (non-parametric) tests.
        Does not account for any interaction or correlation between organisms.
        Results are presented in terms of a single p-value for each organism which
        indicates whether there is a difference between any of the groups.
        """
        _mean_log_rpkm_prefix = "mean_log_rpkm - "
        _median_log_rpkm_prefix = "median_log_rpkm - "

        def primary_args(self):
            return mo.md(
                """
    - {test}
    - {grouping_cname}
    - {query}
                """
            ).batch(
                test=mo.ui.dropdown(
                    label="Statistical Test",
                    options=["T-test", "Mann Whitney U"],
                    value="Mann Whitney U"
                ),
                query=mo.ui.text(
                    label="Filter Specimens (optional):",
                    placeholder="e.g. cname == 'Group A'",
                    full_width=True
                ),
                grouping_cname=mo.ui.dropdown(
                    label="Compare Groups Defined By:",
                    options=self.adata.obs.columns.values,
                    value=(
                        get_grouping_cname()
                        if get_grouping_cname() in self.adata.obs.columns.values
                        else self.adata.obs.columns.values[-1]
                    ),
                    on_change=set_grouping_cname,
                    searchable=True
                )
            )

        def secondary_args(self, grouping_cname: str, query: str, **kwargs):
            try:
                self._filtered_obs(query)
            except ValueError:
                return mo.md(f"Invalid query syntax: {query}").batch()

            # Get the groups present in the selected column
            groups = self._filtered_obs(query)[grouping_cname].unique()

            return mo.md("""
    - {ref_group}
    - {comp_group}
            """).batch(
                ref_group=mo.ui.dropdown(
                    label="Reference Group:",
                    options=groups,
                    value=groups[0],
                    searchable=True
                ),
                comp_group=mo.ui.dropdown(
                    label="Comparison Group:",
                    options=groups,
                    value=groups[1] if len(groups) > 1 else groups[0],
                    searchable=True
                )
            )

        def run_analysis(
            self,
            query: str,
            grouping_cname: str,
            test: str,
            ref_group: str,
            comp_group: str
        ):
            # Make sure that different groups were selected
            if ref_group == comp_group:
                return mo.md("Must select different groups to compare")

            # Optionally filter, and get only those groups which were selected
            obs = self._filtered_obs(query)
            self._groupings = obs.loc[obs[grouping_cname].isin([ref_group, comp_group])][grouping_cname]
            self.ref_group = obs.index.values[obs[grouping_cname] == ref_group]
            self.comp_group = obs.index.values[obs[grouping_cname] == comp_group]

            if test == "T-test":
                self.res = self._compare_groups("ttest_ind")
            elif test == "Mann Whitney U":
                self.res = self._compare_groups("mannwhitneyu")
            else:
                raise Exception(f"Unexpected test {test}")

            self._ready_to_plot = True

        def _compare_groups(self, test_name: str):
            # Compute the results
            df = pd.DataFrame([
                dict(
                    bin=bin,
                    mean_log_rpkm=bin_log_rpkm.mean(),
                    median_log_rpkm=bin_log_rpkm.median(),
                    **self._compare_groups_single(bin_log_rpkm, test_name),
                    **self._group_stats(bin_log_rpkm),
                )
                for bin, bin_log_rpkm in self.log_abund.items()
                if not pd.isnull(bin) and bin != "nan"
            ]).sort_values(by="pvalue")

            # Calculate the log2 fold difference between every pair of groups
            df = df.assign(**{
                f"log2_fold_difference - {cname1[len(self._mean_log_rpkm_prefix):]} / {cname2[len(self._mean_log_rpkm_prefix):]}": df[cname1] - df[cname2]
                for cname1 in df.columns.values
                if cname1.startswith(self._mean_log_rpkm_prefix)
                for cname2 in df.columns
                if cname2.startswith(self._mean_log_rpkm_prefix)
                if cname1 != cname2
            })
            return df

        def _group_stats(self, bin_log_rpkm: pd.Series):
            return {
                kw: val
                for group_name, group_vals in bin_log_rpkm.groupby(self._groupings)
                for kw, val in {
                    self._mean_log_rpkm_prefix + str(group_name): group_vals.mean(),
                    self._median_log_rpkm_prefix + str(group_name): group_vals.median()
                }.items()
            }

        def _compare_groups_single(self, bin_log_rpkm: pd.Series, test_name: str):
            try:
                return dict(zip(
                    ["statistic", "pvalue"],
                    getattr(stats, test_name)(
                        bin_log_rpkm.loc[self.ref_group],
                        bin_log_rpkm.loc[self.comp_group]
                    )
                ))
            except ValueError:
                return dict(statistic=None, pvalue=None)
    return (CompareMetagenomeTwoGroups,)


@app.cell
def class_comparemetagenomemultiplegroups(
    CompareMetagenomeTool,
    List,
    get_grouping_cname,
    mo,
    pd,
    set_grouping_cname,
    stats,
):
    class CompareMetagenomeMultipleGroups(CompareMetagenomeTool):
        name = "Compare Groups - Test Each Organism"
        description = """
        Test each organism individually for differences in abundance between groups.
        Can use either the ANOVA (parametric) or Kruskal-Wallis H (non-parametric) tests.
        Does not account for any interaction or correlation between organisms.
        Results are presented in terms of a single p-value for each organism which
        indicates whether there is a difference between any of the groups.
        """
        _mean_log_rpkm_prefix = "mean_log_rpkm - "
        _median_log_rpkm_prefix = "median_log_rpkm - "

        def primary_args(self):
            return mo.md(
                """
    - {test}
    - {grouping_cname}
    - {query}
                """
            ).batch(
                test=mo.ui.dropdown(
                    label="Statistical Test",
                    options=["Kruskal-Wallis", "ANOVA"],
                    value="Kruskal-Wallis"
                ),
                query=mo.ui.text(
                    label="Filter Specimens (optional):",
                    placeholder="e.g. cname == 'Group A'",
                    full_width=True
                ),
                grouping_cname=mo.ui.dropdown(
                    label="Compare Groups Defined By:",
                    options=self.adata.obs.columns.values,
                    value=get_grouping_cname(),
                    on_change=set_grouping_cname,
                    searchable=True
                )
            )

        def secondary_args(self, grouping_cname: str, query: str, **kwargs):
            try:
                self._filtered_obs(query)
            except ValueError:
                return mo.md(f"Invalid query syntax: {query}").batch()

            # Get the groups present in the selected column
            groups = self._filtered_obs(query)[grouping_cname].unique()

            return mo.md("""
    - {include_groups}
            """).batch(
                include_groups=mo.ui.multiselect(
                    label="Include Groups:",
                    options=groups,
                    value=groups
                )
            )

        def run_analysis(
            self,
            query: str,
            grouping_cname: str,
            test: str,
            include_groups: List[str]
        ):
            # Make sure that multiple groups were included
            if not len(include_groups) > 1:
                return mo.md("Must select multiple groups for analysis")

            # Optionally filter, and get only those groups which were selected
            obs = self._filtered_obs(query)
            self._groupings = obs.loc[
                obs[grouping_cname].isin(include_groups)
            ][grouping_cname]

            if test == "Kruskal-Wallis":
                self.res = self._compare_groups("kruskal")
            elif test == "ANOVA":
                self.res = self._compare_groups("f_oneway")

            self._ready_to_plot = True

        def _compare_groups(self, test_name: str):
            # Compute the results
            df = pd.DataFrame([
                dict(
                    bin=bin,
                    mean_log_rpkm=bin_log_rpkm.mean(),
                    median_log_rpkm=bin_log_rpkm.median(),
                    **self._compare_groups_single(bin_log_rpkm, test_name),
                    **self._group_stats(bin_log_rpkm),
                )
                for bin, bin_log_rpkm in self.log_abund.items()
                if not pd.isnull(bin) and bin != "nan"
            ]).sort_values(by="pvalue")

            # Calculate the log2 fold difference between every pair of groups
            df = df.assign(**{
                f"log2_fold_difference - {cname1[len(self._mean_log_rpkm_prefix):]} / {cname2[len(self._mean_log_rpkm_prefix):]}": df[cname1] - df[cname2]
                for cname1 in df.columns.values
                if cname1.startswith(self._mean_log_rpkm_prefix)
                for cname2 in df.columns
                if cname2.startswith(self._mean_log_rpkm_prefix)
                if cname1 != cname2
            })
            return df

        def _group_stats(self, bin_log_rpkm: pd.Series):
            return {
                kw: val
                for group_name, group_vals in bin_log_rpkm.groupby(self._groupings)
                for kw, val in {
                    self._mean_log_rpkm_prefix + str(group_name): group_vals.mean(),
                    self._median_log_rpkm_prefix + str(group_name): group_vals.median()
                }.items()
            }

        def _compare_groups_single(self, bin_log_rpkm: pd.Series, test_name: str):
            try:
                return dict(zip(
                    ["statistic", "pvalue"],
                    getattr(stats, test_name)(
                        *[
                            vals
                            for _, vals in bin_log_rpkm.groupby(self._groupings)
                        ]
                    )
                ))
            except ValueError:
                return dict(statistic=None, pvalue=None)
    return (CompareMetagenomeMultipleGroups,)


@app.cell
def _(mo):
    with mo.status.spinner("Loading dependencies"):
        from sklearn import ensemble
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        from sklearn.preprocessing import OrdinalEncoder
    return OrdinalEncoder, confusion_matrix, ensemble, train_test_split


@app.cell
def class_comparemetagenometwogroups(
    CompareMetagenomeTool,
    List,
    OrdinalEncoder,
    confusion_matrix,
    ensemble,
    get_grouping_cname,
    mo,
    pd,
    px,
    set_grouping_cname,
    train_test_split,
):
    class CompareMetagenomeClassifyGroupsML(CompareMetagenomeTool):
        name = "Machine Learning – Classify Groups"
        description = """Using the scikit-learn library, build and test
        a machine learning classifier to predict the indicated labels.
        Testing is performed on a randomly selected subset of the data,
        preserving the group balance of the total dataset.

        Outputs will indicate the importance of each organism in the
        model, as well as the predictive accuracy and confusion matrix.
        """

        _options=[
            fname
            for fname in dir(ensemble)
            if fname.endswith("Classifier") and fname not in ["VotingClassifier", "StackingClassifier"]
        ]

        def primary_args(self):
            return mo.md(
                """
    - {classifier}
    - {training_prop}
    - {query}
    - {grouping_cname}
                """
            ).batch(
                classifier=mo.ui.dropdown(
                    label="Classifier:",
                    options=self._options,
                    value="RandomForestClassifier",
                    searchable=True
                ),
                training_prop=mo.ui.number(
                    label="Training Proportion:",
                    start=0.1,
                    stop=0.9,
                    step=0.01,
                    value=0.5
                ),
                query=mo.ui.text(
                    label="Filter Specimens (optional):",
                    placeholder="e.g. cname == 'Group A'",
                    full_width=True
                ),
                grouping_cname=mo.ui.dropdown(
                    label="Compare Groups Defined By:",
                    options=self.adata.obs.columns.values,
                    value=get_grouping_cname(),
                    on_change=set_grouping_cname,
                    searchable=True
                )
            )

        def secondary_args(self, grouping_cname: str, query: str, **kwargs):
            try:
                self._filtered_obs(query)
            except ValueError:
                return mo.md(f"Invalid query syntax: {query}").batch()

            # Get the groups present in the selected column
            groups = self._filtered_obs(query)[grouping_cname].unique()

            return mo.md("""
    - {include_groups}
            """).batch(
                include_groups=mo.ui.multiselect(
                    label="Include Groups:",
                    options=groups,
                    value=groups
                )
            )

        def run_analysis(
            self,
            query: str,
            grouping_cname: str,
            include_groups: List[str],
            classifier: str,
            training_prop: float
        ):
            # Make sure that multiple groups were included
            if not len(include_groups) > 1:
                return mo.md("Must select multiple groups for analysis")

            # Optionally filter, and get only those groups which were selected
            obs = self._filtered_obs(query)
            self._groupings = obs.loc[
                obs[grouping_cname].isin(include_groups),
                [grouping_cname]
            ]

            # Encode the groups as numeric values for sklearn to work with
            self.onc = OrdinalEncoder()
            self.y = self.onc.fit_transform(self._groupings)

            # Encode the abundances as numeric
            self.X = self.log_abund.reindex(index=self._groupings.index).apply(pd.to_numeric)

            # Instantiate the ML
            self.model = getattr(ensemble, classifier)()

            # Get the training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                train_size=training_prop,
                stratify=self.y
            )

            # Fit the model
            self.model.fit(self.X_train, self.y_train)

            # Make predictions
            self.pred_test = self.model.predict(self.X_test)
            self.pred_proba_test = [
                proba_arr[int(pred_ix)]
                for proba_arr, pred_ix in zip(
                    self.model.predict_proba(self.X_test),
                    self.pred_test
                )
            ]

            # Make a DataFrame with all of the prediction information
            self.pred_df = pd.DataFrame(dict(
                truth=self.onc.inverse_transform(self.y_test)[:, 0],
                prediction=self.onc.inverse_transform(pd.DataFrame(self.pred_test))[:, 0],
                probability=self.pred_proba_test
            )).assign(
                correct=lambda df: df["truth"] == df["prediction"]
            )

            self.score = self.model.score(self.X_test, self.y_test)

            if hasattr(self.model, "feature_importances_"):
                self.res = pd.DataFrame(
                    dict(
                        bin=self.log_abund.columns,
                        feature_importance=self.model.feature_importances_
                    )
                ).sort_values(
                    by="feature_importance",
                    ascending=False
                )
                self.feature_importances = self.res.set_index("bin")["feature_importance"]

                self.feature_importances = self.feature_importances.loc[self.feature_importances > 0]

                # Feature Importance Figure
                feature_importance_fig = px.bar(
                    self.feature_importances,
                    title="Feature Importances",
                    template="simple_white"
                )
                feature_importance_fig.update_layout(showlegend=False)

            else:
                self.res = None
                self.feature_importances = None
                feature_importance_fig = mo.md("")

            # Confusion Matrix Figure
            self.conf_mat = pd.DataFrame(
                confusion_matrix(self.pred_df["truth"], self.pred_df["prediction"]),
                index=self.pred_df["truth"].drop_duplicates().sort_values().values,
                columns=self.pred_df["truth"].drop_duplicates().sort_values().values,
            )
            conf_mat_fig = px.imshow(
                self.conf_mat.values,
                zmin=0,
                x=self.conf_mat.index.values,
                y=self.conf_mat.index.values,
                color_continuous_scale="gray_r",
                title="Confusion Matrix",
                template="simple_white",
                labels=dict(
                    x="Predicted Group",
                    y="Actual Group"
                )
            )

            # Accuracy vs. prediction probability figure
            proba_df = pd.DataFrame([
                dict(
                    probability=probability,
                    accuracy=self.pred_df.query(f"probability >= {probability}")["correct"].mean()
                )
                for probability in self.pred_df["probability"].unique()
            ]).sort_values(by="probability")
            proba_fig = px.line(
                data_frame=proba_df,
                x="probability",
                y="accuracy",
                template="simple_white",
                labels=dict(
                    probability="Prediction Probability Threshold",
                    accuracy="Prediction Accuracy"
                )
            )

            return mo.vstack([
                mo.md(f"### {classifier}\n\n- Overall Prediction Accuracy: {100 * self.score:.2f}%"),
                conf_mat_fig,
                feature_importance_fig,
                proba_fig
            ])

        def _train_test_split(self, training_prop: float):
            """Generate the training and testing datasets."""

            train_idx = [
                ix
                for group, group_vals in self._groupings.groupby(self._groupings)
                for ix in group_vals.sample(int(group_vals.shape[0] * training_prop)).index.values
            ]
            test_idx = [
                ix
                for ix in self._groupings.index.values
                if ix not in train_idx
            ]
            return self._groupings.reindex(train_idx), self.log_abund.reindex(index=train_idx), self._groupings.reindex(test_idx), self.log_abund.reindex(index=test_idx)

    return (CompareMetagenomeClassifyGroupsML,)


@app.cell
def _(
    CompareMetagenomeClassifyGroupsML,
    CompareMetagenomeMultipleGroups,
    CompareMetagenomeTool,
    CompareMetagenomeTwoGroups,
    List,
    mo,
):
    class CompareMetagenome:
        tools = List[CompareMetagenomeTool]

        def __init__(self):
            self.tools = [
                CompareMetagenomeTwoGroups,
                CompareMetagenomeMultipleGroups,
                CompareMetagenomeClassifyGroupsML,
            ]

        def analysis_type_args(self):
            return mo.ui.dropdown(
                label="Analysis Type:",
                options=[
                    tool.name
                    for tool in self.tools                    
                ],
                value=self.tools[0].name,
                searchable=True
            )

        def make_tool(self, tool_name: str):
            for tool in self.tools:
                if tool.name == tool_name:
                    return tool()
            raise ValueError(f"No such tool: {tool_name}")


    compare_metagenome = CompareMetagenome()
    return (compare_metagenome,)


@app.cell
def _(compare_metagenome):
    compare_metagenome_analysis_type = compare_metagenome.analysis_type_args()
    compare_metagenome_analysis_type
    return (compare_metagenome_analysis_type,)


@app.cell
def _(compare_metagenome, compare_metagenome_analysis_type, mo):
    # Instantiate the tool selected by the user
    compare_metagenome_tool = compare_metagenome.make_tool(compare_metagenome_analysis_type.value)
    mo.md(compare_metagenome_tool.description)
    return (compare_metagenome_tool,)


@app.cell
def _():
    return


@app.cell
def _(compare_metagenome_tool):
    compare_metagenome_primary_args = compare_metagenome_tool.primary_args()
    compare_metagenome_primary_args
    return (compare_metagenome_primary_args,)


@app.cell
def _(compare_metagenome_primary_args, compare_metagenome_tool):
    compare_metagenome_secondary_args = compare_metagenome_tool.secondary_args(**compare_metagenome_primary_args.value)
    compare_metagenome_secondary_args
    return (compare_metagenome_secondary_args,)


@app.cell
def _(
    compare_metagenome_primary_args,
    compare_metagenome_secondary_args,
    compare_metagenome_tool,
):
    compare_metagenome_tool.run_analysis(
        **compare_metagenome_primary_args.value,
        **compare_metagenome_secondary_args.value
    )
    return


@app.cell
def _(
    compare_metagenome_primary_args,
    compare_metagenome_secondary_args,
    compare_metagenome_tool,
):
    compare_metagenome_tool_primary_plot_args = compare_metagenome_tool.primary_plot_args(
        **compare_metagenome_primary_args.value,
        **compare_metagenome_secondary_args.value
    )
    compare_metagenome_tool_primary_plot_args
    return (compare_metagenome_tool_primary_plot_args,)


@app.cell
def _(
    compare_metagenome_primary_args,
    compare_metagenome_secondary_args,
    compare_metagenome_tool,
    compare_metagenome_tool_primary_plot_args,
    mo,
):
    mo.stop(compare_metagenome_tool._ready_to_plot is False)
    compare_metagenome_tool.make_primary_plot(
        **compare_metagenome_tool_primary_plot_args.value,
        **compare_metagenome_primary_args.value,
        **compare_metagenome_secondary_args.value
    )
    return


@app.cell
def _(
    compare_metagenome_primary_args,
    compare_metagenome_secondary_args,
    compare_metagenome_tool,
    compare_metagenome_tool_primary_plot_args,
):
    compare_metagenome_tool_secondary_plot_args = compare_metagenome_tool.secondary_plot_args(
        **compare_metagenome_tool_primary_plot_args.value,
        **compare_metagenome_primary_args.value,
        **compare_metagenome_secondary_args.value
    )
    compare_metagenome_tool_secondary_plot_args
    return (compare_metagenome_tool_secondary_plot_args,)


@app.cell
def _(
    compare_metagenome_primary_args,
    compare_metagenome_secondary_args,
    compare_metagenome_tool,
    compare_metagenome_tool_primary_plot_args,
    compare_metagenome_tool_secondary_plot_args,
):
    compare_metagenome_tool.make_secondary_plot(
        **compare_metagenome_tool_primary_plot_args.value,
        **compare_metagenome_tool_secondary_plot_args.value,
        **compare_metagenome_primary_args.value,
        **compare_metagenome_secondary_args.value
    )
    return


@app.cell
def _(
    compare_metagenome_primary_args,
    compare_metagenome_secondary_args,
    compare_metagenome_tool,
    compare_metagenome_tool_primary_plot_args,
    compare_metagenome_tool_secondary_plot_args,
):
    compare_metagenome_tool_tertiary_plot_args = compare_metagenome_tool.tertiary_plot_args(
        **compare_metagenome_tool_primary_plot_args.value,
        **compare_metagenome_tool_secondary_plot_args.value,
        **compare_metagenome_primary_args.value,
        **compare_metagenome_secondary_args.value
    )
    compare_metagenome_tool_tertiary_plot_args
    return (compare_metagenome_tool_tertiary_plot_args,)


@app.cell
def _(
    compare_metagenome_primary_args,
    compare_metagenome_secondary_args,
    compare_metagenome_tool,
    compare_metagenome_tool_primary_plot_args,
    compare_metagenome_tool_secondary_plot_args,
    compare_metagenome_tool_tertiary_plot_args,
):
    compare_metagenome_tool.make_tertiary_plot(
        **compare_metagenome_tool_primary_plot_args.value,
        **compare_metagenome_tool_secondary_plot_args.value,
        **compare_metagenome_tool_tertiary_plot_args.value,
        **compare_metagenome_primary_args.value,
        **compare_metagenome_secondary_args.value

    )
    return


@app.cell
def _(compare_metagenome_primary_args, compare_metagenome_secondary_args, mo):
    compare_metagenome_primary_args.value
    compare_metagenome_secondary_args.value

    run_calc_gene_annot_button = mo.ui.run_button(
        label="Calculate Gene Annotation Enrichment"
    )
    run_calc_gene_annot_button
    return (run_calc_gene_annot_button,)


@app.cell
def _(compare_metagenome_tool, mo, run_calc_gene_annot_button):
    if run_calc_gene_annot_button.value:
        with mo.status.spinner("Calculating Gene Annotation Enrichment"):
            gene_annot_df = compare_metagenome_tool.calc_gene_annot_enrichment()
    else:
        gene_annot_df = None
    return (gene_annot_df,)


@app.cell
def gene_annot_enrichment_args(compare_metagenome_tool, gene_annot_df, mo):
    mo.stop(gene_annot_df is None)
    compare_metagenome_tool_gene_annot_args = compare_metagenome_tool.gene_annot_enrichment_args()
    compare_metagenome_tool_gene_annot_args
    return (compare_metagenome_tool_gene_annot_args,)


@app.cell
def _(
    compare_metagenome_tool,
    compare_metagenome_tool_gene_annot_args,
    gene_annot_df,
):
    compare_metagenome_tool.gene_annot_enrichment_plot(
        gene_annot_df,
        **compare_metagenome_tool_gene_annot_args.value
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Settings""")
    return


@app.cell
def _(mo, query_params):
    mo.md("""
    ---

    The gig-map analysis will identify every individual gene which is encoded by each genome
    in the pan-genome collection.
    To determine whether a particular genome contains a particular gene bin (which generally consist
    of multiple genes), a minimum threshold is applied for the proportion of genes which must be detected
    for that bin to be counted as present.

    {min_prop}

    ---

    Genomes with highly fragmented assemblies may not contain a large number of completely-assembled
    genes from the pan-genome collection.
    To filter out these genomes, a lower bound is set on the number of genes that are contained in a
    genome.
    This threshold is set as a proportion relative to the median number of genes encoded by genomes
    in this dataset.

    {min_genes_rel_median}

    ---
    """).batch(
        min_prop=mo.ui.number(
            label="Minimum Proportion of Genes in Bin:",
            start=0.01,
            stop=1.0,
            value=float(query_params.get("min_prop", '0.5'))
        ),
        min_genes_rel_median=mo.ui.number(
            label="Minimum Number of Genes per Genome (Median Fraction):",
            start=0.01,
            stop=1.0,
            step=0.01,
            value=float(query_params.get("min_genes_rel_median", '0.5'))
        )
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
