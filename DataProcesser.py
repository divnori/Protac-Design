"""
The `DataProcesser` class contains functions for pre-processing training data.
"""
# load general packages and functions
import os
import numpy as np
import rdkit
import h5py
from tqdm import tqdm

# load GraphINVENT-specific functions
from Analyzer import Analyzer
from parameters.constants import constants
import parameters.load as load
from MolecularGraph import PreprocessingGraph
import util


class DataProcesser:
    """
    A class for preprocessing molecular sets and writing them to HDF files.
    """
    def __init__(self, path : str, is_training_set : bool=False, molset = []) -> None:
        """
        Args:
        ----
            path (string)          : Full path/filename to SMILES file containing
                                     molecules.
            is_training_set (bool) : Indicates if this is the training set, as we
                                     calculate a few additional things for the training
                                     set.
        """
        # define some variables for later use
        self.path            = path
        print("path defined", flush = True)
        self.is_training_set = is_training_set
        print("training set defined", flush = True)
        self.dataset_names   = ["nodes", "edges", "APDs"]
        print("dataset names defined", flush = True)
        self.get_dataset_dims()  # creates `self.dims`
        print("get dims done", flush = True)

        # load the molecules
        if constants.compute_train_csv:
            self.molecule_set = molset
        else:
            self.molecule_set = load.molecules(self.path)
        print("molecules loaded", flush = True)
        print(f"len of loaded molecules {len(self.molecule_set)}", flush=True)
        #self.molecule_set = molset

        # placeholders
        self.molecule_subset    = None
        self.dataset            = None
        self.skip_collection    = None
        self.resume_idx         = None
        self.ts_properties      = None
        self.restart_index_file = None
        self.hdf_file           = None
        self.dataset_size       = None

        print("placeholders set", flush = True)

        # get total number of molecules, and total number of subgraphs in their
        # decoding routes
        self.n_molecules       = len(self.molecule_set)
        print("got len", flush = True)
        self.total_n_subgraphs = self.get_n_subgraphs()
        print("get subgraphs done", flush = True)
        print(f"-- {self.n_molecules} molecules in set.", flush=True)
        print(f"-- {self.total_n_subgraphs} total subgraphs in set.",
              flush=True)

    def preprocess(self) -> None:
        """
        Prepares an HDF file to save three different datasets to it (`nodes`,
        `edges`, `APDs`), and slowly fills it in by looping over all the
        molecules in the data in groups (or "mini-batches").
        """
        with h5py.File(f"{self.path[:-3]}h5.chunked", "a") as self.hdf_file:

            self.restart_index_file = constants.dataset_dir + "index.restart"

            if constants.restart and os.path.exists(self.restart_index_file):
                self.restart_preprocessing_job()
            else:
                self.start_new_preprocessing_job()

                # keep track of the dataset size (to resize later)
                self.dataset_size = 0

            self.ts_properties = None

            # this is where we fill the datasets with actual data by looping
            # over subgraphs in blocks of size `constants.batch_size`
            for idx in range(0, self.total_n_subgraphs, constants.batch_size):

                if not self.skip_collection:

                    # add `constants.batch_size` subgraphs from
                    # `self.molecule_subset` to the dataset (and if training
                    # set, calculate their properties and add these to
                    # `self.ts_properties`)
                    self.get_subgraphs(init_idx=idx)

                    util.write_last_molecule_idx(
                        last_molecule_idx=self.resume_idx,
                        dataset_size=self.dataset_size,
                        restart_file_path=constants.dataset_dir
                    )


                if self.resume_idx == self.n_molecules:
                    # all molecules have been processed

                    self.resize_datasets()  # remove padding from initialization
                    print("Datasets resized.", flush=True)

                    print(f"mol set length is {len(self.molecule_set)}",flush=True)

                    if self.is_training_set and not constants.restart:
                        print("Writing training set properties.", flush=True)
                        util.write_ts_properties(
                            training_set_properties=self.ts_properties
                        )

                    break

        print("* Resaving datasets in unchunked format.")
        self.resave_datasets_unchunked()

    def restart_preprocessing_job(self) -> None:
        """
        Restarts a preprocessing job. Uses an index specified in the dataset
        directory to know where to resume preprocessing.
        """
        try:
            self.resume_idx, self.dataset_size = util.read_last_molecule_idx(
                restart_file_path=constants.dataset_dir
            )
        except:
            self.resume_idx, self.dataset_size = 0, 0
        self.skip_collection = bool(
            self.resume_idx == self.n_molecules and self.is_training_set
        )

        # load dictionary of previously created datasets (`self.dataset`)
        self.load_datasets(hdf_file=self.hdf_file)

    def start_new_preprocessing_job(self) -> None:
        """
        Starts a fresh preprocessing job.
        """
        self.resume_idx      = 0
        self.skip_collection = False

        # create a dictionary of empty HDF datasets (`self.dataset`)
        self.create_datasets(hdf_file=self.hdf_file)

    def resave_datasets_unchunked(self) -> None:
        """
        Resaves the HDF datasets in an unchunked format to remove initial
        padding.
        """
        with h5py.File(f"{self.path[:-3]}h5.chunked", "r", swmr=True) as chunked_file:
            keys        = list(chunked_file.keys())
            data        = [chunked_file.get(key)[:] for key in keys]
            data_zipped = tuple(zip(data, keys))

            with h5py.File(f"{self.path[:-3]}h5", "w") as unchunked_file:
                for d, k in tqdm(data_zipped):
                    unchunked_file.create_dataset(
                        k, chunks=None, data=d, dtype=np.dtype("int8")
                    )

        # remove the restart file and chunked file (don't need them anymore)
        os.remove(self.restart_index_file)
        os.remove(f"{self.path[:-3]}h5.chunked")

    def get_subgraphs(self, init_idx = 0) -> None:
        """
        Adds `constants.batch_size` subgraphs from `self.molecule_subset` to the
        HDF dataset (and if currently processing the training set, also
        calculates the full graphs' properties and adds these to
        `self.ts_properties`).

        Args:
        ----
            init_idx (int) : As analysis is done in blocks/slices, `init_idx` is
                             the start index for the next block/slice to be taken
                             from `self.molecule_subset`.
        """
        data_subgraphs, data_apds, molecular_graph_list = [], [], []  # initialize

        # convert all molecules in `self.molecules_subset` to `PreprocessingGraphs`
        molecular_graph_generator = map(self.get_graph, self.molecule_subset)
        molecular_graph_generator = map(self.get_graph, self.molecule_set)

        molecules_processed       = 0  # keep track of the number of molecules processed

        # # loop over all the `PreprocessingGraph`s
        for graph in molecular_graph_generator:
            molecules_processed += 1

            # store `PreprocessingGraph` object
            molecular_graph_list.append(graph)

            # get the number of decoding graphs
            n_subgraphs = graph.get_decoding_route_length()

            for new_subgraph_idx in range(n_subgraphs):

                # `get_decoding_route_state() returns a list of [`subgraph`, `apd`],
                subgraph, apd = graph.get_decoding_route_state(
                    subgraph_idx=new_subgraph_idx
                )

                # "collect" all APDs corresponding to pre-existing subgraphs,
                # otherwise append both new subgraph and new APD
                count = 0
                for idx, existing_subgraph in enumerate(data_subgraphs):

                    count += 1
                    # check if subgraph `subgraph` is "already" in
                    # `data_subgraphs` as `existing_subgraph`, and if so, add
                    # the "new" APD to the "old"
                    try:  # first compare the node feature matrices
                        nodes_equal = (subgraph[0] == existing_subgraph[0]).all()
                    except AttributeError:
                        nodes_equal = False
                    try:  # then compare the edge feature tensors
                        edges_equal = (subgraph[1] == existing_subgraph[1]).all()
                    except AttributeError:
                        edges_equal = False

                    # if both matrices have a match, then subgraphs are the same
                    if nodes_equal and edges_equal:
                        existing_apd = data_apds[idx]
                        existing_apd += apd
                        break

                # if subgraph is not already in `data_subgraphs`, append it
                if count == len(data_subgraphs) or count == 0:
                    data_subgraphs.append(subgraph)
                    data_apds.append(apd)

                # if `constants.batch_size` unique subgraphs have been
                # processed, save group to the HDF dataset
                len_data_subgraphs = len(data_subgraphs)
                if len_data_subgraphs == constants.batch_size:
                    self.save_group(data_subgraphs=data_subgraphs,
                                    data_apds=data_apds,
                                    group_size=len_data_subgraphs,
                                    init_idx=init_idx)

                    # get molecular properties for group iff it's the training set
                    self.get_ts_properties(molecular_graphs=molecular_graph_list,
                                           group_size=constants.batch_size)

                    # keep track of the last molecule to be processed in
                    # `self.resume_idx`
                    # number of molecules processed:
                    self.resume_idx   += molecules_processed
                    # subgraphs processed:
                    self.dataset_size += constants.batch_size

                    return None

        n_processed_subgraphs = len(data_subgraphs)

        # save group with < `constants.batch_size` subgraphs (e.g. last block)
        self.save_group(data_subgraphs=data_subgraphs,
                        data_apds=data_apds,
                        group_size=n_processed_subgraphs,
                        init_idx=init_idx)

        # get molecular properties for this group iff it's the training set
        self.get_ts_properties(molecular_graphs=molecular_graph_list,
                               group_size=constants.batch_size)

        # # keep track of the last molecule to be processed in `self.resume_idx`
        # self.resume_idx   += molecules_processed  # number of molecules processed
        # self.dataset_size += molecules_processed  # subgraphs processed

        return None

    def create_datasets(self, hdf_file : h5py._hl.files.File) -> None:
        """
        Creates a dictionary of HDF5 datasets (`self.dataset`).

        Args:
        ----
            hdf_file (h5py._hl.files.File) : HDF5 file which will contain the datasets.
        """
        self.dataset = {}  # initialize

        for ds_name in self.dataset_names:
            self.dataset[ds_name] = hdf_file.create_dataset(
                ds_name,
                (self.total_n_subgraphs, *self.dims[ds_name]),
                chunks=True,  # must be True for resizing later
                dtype=np.dtype("int8")
            )

    def resize_datasets(self) -> None:
        """
        Resizes the HDF datasets, since much longer datasets are initialized
        when first creating the HDF datasets (it it is impossible to predict
        how many graphs will be equivalent beforehand).
        """
        for dataset_name in self.dataset_names:
            try:
                self.dataset[dataset_name].resize(
                    (self.dataset_size, *self.dims[dataset_name]))
            except KeyError:  # `f_term` has no extra dims
                self.dataset[dataset_name].resize((self.dataset_size,))

    def get_dataset_dims(self) -> None:
        """
        Calculates the dimensions of the node features, edge features, and APDs,
        and stores them as lists in a dict (`self.dims`), where keys are the
        dataset name.

        Shapes:
        ------
            dims["nodes"] : [max N nodes, N atom types + N formal charges]
            dims["edges"] : [max N nodes, max N nodes, N bond types]
            dims["APDs"]  : [APD length = f_add length + f_conn length + f_term length]
        """
        self.dims = {}
        self.dims["nodes"] = constants.dim_nodes
        self.dims["edges"] = constants.dim_edges
        self.dims["APDs"]  = constants.dim_apd

    def get_graph(self, mol : rdkit.Chem.Mol) -> PreprocessingGraph:
        """
        Converts an `rdkit.Chem.Mol` object to `PreprocessingGraph`.

        Args:
        ----
            mol (rdkit.Chem.Mol) : Molecule to convert.

        Returns:
        -------
            molecular_graph (PreprocessingGraph) : Molecule, now as a graph.
        """
        if mol is not None:
            print(rdkit.Chem.MolToSmiles(mol),flush=True)
            if not constants.use_aromatic_bonds:
                rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
            molecular_graph = PreprocessingGraph(molecule=mol,
                                                 constants=constants)
        return molecular_graph

    def get_molecule_subset(self) -> None:
        """
        Slices `self.molecule_set` into a subset of molecules of size
        `constants.batch_size`, starting from `self.resume_idx`.
        `self.n_molecules` is the number of molecules in the full
        `self.molecule_set`.
        """
        init_idx             = self.resume_idx
        subset_size          = constants.batch_size
        self.molecule_subset = []
        max_idx              = min(init_idx + subset_size, self.n_molecules)

        count = -1
        for mol in self.molecule_set:
            if mol is not None:
                count += 1
                if count < init_idx:
                    continue
                elif count >= max_idx:
                    return self.molecule_subset
                else:
                    self.molecule_subset.append(mol)

    def get_n_subgraphs(self) -> int:
        """
        Calculates the total number of subgraphs in the decoding route of all
        molecules in `self.molecule_set`. Loads training, testing, or validation
        set. First, the `PreprocessingGraph` for each molecule is obtained, and
        then the length of the decoding route is trivially calculated for each.

        Returns:
        -------
            n_subgraphs (int) : Sum of number of subgraphs in decoding routes for
                                all molecules in `self.molecule_set`.
        """
        n_subgraphs = 0  # start the count
        print("initialized n_subgraphs",flush=True)

        # convert molecules in `self.molecule_set` to `PreprocessingGraph`s
        molecular_graph_generator = map(self.get_graph, self.molecule_set)
        print("converted mol set to graph",flush=True)

        # loop over all the `PreprocessingGraph`s
        for molecular_graph in molecular_graph_generator:
            # get the number of decoding graphs (i.e. the decoding route length)
            # and add them to the running count
            n_subgraphs += molecular_graph.get_decoding_route_length()
            print("got route length",flush=True)

        print("done w graph loop",flush=True)

        return int(n_subgraphs)

    def get_ts_properties(self, molecular_graphs : list, group_size : int) -> \
        None:
        """
        Gets molecular properties for group of molecular graphs, only for the
        training set.

        Args:
        ----
            molecular_graphs (list) : Contains `PreprocessingGraph`s.
            group_size (int)        : Size of "group" (i.e. slice of graphs).
        """
        if self.is_training_set:
            print("is training set so calculating training set props", flush=True)

            analyzer      = Analyzer()
            ts_properties = analyzer.evaluate_training_set(
                preprocessing_graphs=molecular_graphs
            )
            print("initialized ts analyzer",flush=True)

            # merge properties of current group with the previous group analyzed
            if self.ts_properties:  # `self.ts_properties` is a dictionary
                print("in self.ts_properties conditional", flush=True)
                if not constants.compute_train_csv:
                    self.ts_properties = analyzer.combine_ts_properties(
                        prev_properties=self.ts_properties,
                        next_properties=ts_properties,
                        weight_next=group_size
                    )
                else:
                    print("setting self.ts_properties", flush=True)
                    self.ts_properties = ts_properties
            else:  # `self.ts_properties` is None (has not been calculated yet)
                self.ts_properties = ts_properties
        else:
            self.ts_properties = None
        print(self.ts_properties)
        

    def load_datasets(self, hdf_file : h5py._hl.files.File) -> None:
        """
        Creates a dictionary of HDF datasets (`self.dataset`) which have been
        previously created (for restart jobs only).

        Args:
        ----
            hdf_file (h5py._hl.files.File) : HDF file containing all the datasets.
        """
        self.dataset = {}  # initialize dictionary of datasets

        # use the names of the datasets as the keys in `self.dataset`
        for ds_name in self.dataset_names:
            self.dataset[ds_name] = hdf_file.get(ds_name)

    def save_group(self, data_subgraphs : list, data_apds : list,
                   group_size : int, init_idx : int) -> None:
        """
        Saves a group of padded subgraphs and their corresponding APDs to the HDF
        datasets as `numpy.ndarray`s.

        Args:
        ----
            data_subgraphs (list) : Contains molecular subgraphs.
            data_apds (list)      : Contains APDs.
            group_size (int)      : Size of HDF "slice".
            init_idx (int)        : Index to begin slicing.
        """
        # convert to `np.ndarray`s
        nodes = np.array([graph_tuple[0] for graph_tuple in data_subgraphs])
        edges = np.array([graph_tuple[1] for graph_tuple in data_subgraphs])
        apds  = np.array(data_apds)

        end_idx = init_idx + group_size  # idx to end slicing

        # once data is padded, save it to dataset slice
        self.dataset["nodes"][init_idx:end_idx] = nodes
        self.dataset["edges"][init_idx:end_idx] = edges
        self.dataset["APDs"][init_idx:end_idx]  = apds
