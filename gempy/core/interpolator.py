from gempy.core.data import SurfacePoints, Orientations, Grid, Surfaces, Series, Faults, AdditionalData
from gempy.utils.meta import setdoc_pro, setdoc
import gempy.utils.docstring as ds

import numpy as np
import theano


@setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, Grid.__doc__, Surfaces.__doc__, Series.__doc__,
             Faults.__doc__, AdditionalData.__doc__])
class Interpolator(object):
    """Class that act as:
     1) linker between the data objects and the theano graph
     2) container of theano graphs + shared variables
     3) container of theano function

    Args:
        surface_points (SurfacePoints): [s0]
        orientations (Orientations): [s1]
        grid (Grid): [s2]
        surfaces (Surfaces): [s3]
        series (Series): [s4]
        faults (Faults): [s5]
        additional_data (AdditionalData): [s6]
        kwargs:
            - compile_theano: if true, the function is compile at the creation of the class

    Attributes:
        surface_points (SurfacePoints)
        orientations (Orientations)
        grid (Grid)
        surfaces (Surfaces)
        faults (Faults)
        additional_data (AdditionalData)
        dtype (['float32', 'float64']): float precision
        theano_graph: theano graph object with the properties from AdditionalData -> Options
        theano function: python function to call the theano code

    """
    # TODO assert passed data is rescaled

    def __init__(self, surface_points: "SurfacePoints", orientations: "Orientations", grid: "Grid",
                 surfaces: "Surfaces", series: Series, faults: "Faults", additional_data: "AdditionalData", **kwargs):

        self.surface_points = surface_points
        self.orientations = orientations
        self.grid = grid
        self.additional_data = additional_data
        self.surfaces = surfaces
        self.series = series
        self.faults = faults

        self.dtype = additional_data.options.df.loc['values', 'dtype']
        self.theano_graph = self.create_theano_graph(additional_data, inplace=False)
        self.theano_function = None

        self._compute_len_series()

    def _compute_len_series(self):
        self.len_series_i = self.additional_data.structure_data.df.loc['values', 'len series surface_points'] - \
                            self.additional_data.structure_data.df.loc['values', 'number surfaces per series']
        if self.len_series_i.shape[0] == 0:
            self.len_series_i = np.zeros(1, dtype=int)

        self.len_series_o = self.additional_data.structure_data.df.loc['values', 'len series orientations'].astype(
            'int32')
        if self.len_series_o.shape[0] == 0:
            self.len_series_o = np.zeros(1, dtype=int)

        self.len_series_u = self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype('int32')
        if self.len_series_u.shape[0] == 0:
            self.len_series_u = np.zeros(1, dtype=int)

        self.len_series_f = self.faults.faults_relations_df.sum(axis=0).values.astype('int32')[
                            :self.additional_data.get_additional_data()['values']['Structure', 'number series']]
        if self.len_series_f.shape[0] == 0:
            self.len_series_f = np.zeros(1, dtype=int)

        self.len_series_w = self.len_series_i + self.len_series_o * 3 + self.len_series_u + self.len_series_f

    @setdoc_pro([AdditionalData.__doc__, ds.inplace, ds.theano_graph_pro])
    def create_theano_graph(self, additional_data: "AdditionalData" = None, inplace=True):
        """
        Create the graph accordingly to the options in the AdditionalData object

        Args:
            additional_data (AdditionalData): [s0]
            inplace (bool): [s1]

        Returns:
            TheanoGraphPro: [s2]
        """
        import gempy.core.theano.theano_graph_pro as tg
        import importlib
        importlib.reload(tg)

        if additional_data is None:
            additional_data = self.additional_data

        graph = tg.TheanoGraphPro(optimizer=additional_data.options.df.loc['values', 'theano_optimizer'],
                                  dtype=additional_data.options.df.loc['values', 'dtype'],
                                  verbose=additional_data.options.df.loc['values', 'verbosity'])
        if inplace is True:
            self.theano_graph = graph
        else:
            return graph

    @setdoc_pro([ds.theano_graph_pro])
    def set_theano_graph(self, th_graph):
        """
        Attach an already create theano graph.

        Args:
            th_graph (TheanoGraphPro): [s0]

        Returns:
            True
        """
        self.theano_graph = th_graph
        return True

    def set_theano_shared_kriging(self):
        """
        Set to the theano_graph attribute the shared variables of kriging values from the linked
         :class:`AdditionalData`.

        Returns:
            True
        """
        # Range
        # TODO add rescaled range and co into the rescaling data df?
        self.theano_graph.a_T.set_value(np.cast[self.dtype]
                                        (self.additional_data.kriging_data.df.loc['values', 'range'] /
                                         self.additional_data.rescaling_data.df.loc[
                                             'values', 'rescaling factor']))
        # Covariance at 0
        self.theano_graph.c_o_T.set_value(np.cast[self.dtype](
            self.additional_data.kriging_data.df.loc['values', '$C_o$'] /
            self.additional_data.rescaling_data.df.loc[
                'values', 'rescaling factor']
        ))
        # universal grades
        self.theano_graph.n_universal_eq_T.set_value(
            list(self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype('int32')))

        self.set_theano_shared_nuggets()

    def set_theano_shared_nuggets(self):
        # nugget effect
        # len_orientations = self.additional_data.structure_data.df.loc['values', 'len series orientations']
        # len_orientations_len = np.sum(len_orientations)

        self.theano_graph.nugget_effect_grad_T.set_value(
            np.cast[self.dtype](np.tile(
                self.orientations.df['smooth'], 3)))

        # len_rest_form = (self.additional_data.structure_data.df.loc['values', 'len surfaces surface_points'])
        # len_rest_len = np.sum(len_rest_form)
        self.theano_graph.nugget_effect_scalar_T.set_value(
            np.cast[self.dtype](self.surface_points.df['smooth']))
        return True

    def set_theano_shared_structure_surfaces(self):
        """
        Set to the theano_graph attribute the shared variables of structure from the linked
         :class:`AdditionalData`.

        Returns:
            True
        """
        len_rest_form = (self.additional_data.structure_data.df.loc['values', 'len surfaces surface_points'] - 1)
        self.theano_graph.number_of_points_per_surface_T.set_value(len_rest_form.astype('int32'))


class InterpolatorWeights(Interpolator):

    def __init__(self, surface_points: "SurfacePoints", orientations: "Orientations", grid: "Grid",
                 surfaces: "Surfaces", series, faults: "Faults", additional_data: "AdditionalData", **kwargs):

        super(InterpolatorWeights, self).__init__(surface_points, orientations, grid, surfaces, series, faults,
                                                  additional_data, **kwargs)

    def get_python_input_weights(self, fault_drift=None):
        """
             Get values from the data objects used during the interpolation:
                 - dip positions XYZ
                 - dip angles
                 - azimuth
                 - polarity
                 - surface_points coordinates XYZ
             Returns:
                 (list)
             """
        # orientations, this ones I tile them inside theano. PYTHON VAR
        dips_position = self.orientations.df[['X_r', 'Y_r', 'Z_r']].values
        dip_angles = self.orientations.df["dip"].values
        azimuth = self.orientations.df["azimuth"].values
        polarity = self.orientations.df["polarity"].values
        surface_points_coord = self.surface_points.df[['X_r', 'Y_r', 'Z_r']].values
        if fault_drift is None:
            fault_drift = np.zeros((0, self.grid.values.shape[0] + 2 * self.len_series_i.sum()))

          #  fault_drift = np.zeros((0, surface_points_coord.shape[0]))

        # Set all in a list casting them in the chosen dtype
        idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity, surface_points_coord,
                                                  fault_drift)]
        return idl

    def compile_th_fn(self, inplace=False, debug=False):

        self.set_theano_shared_kriging()
        self.set_theano_shared_structure_surfaces()
        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.theano_graph.input_parameters_kriging
        print('Compiling theano function...')

        th_fn = theano.function(input_data_T,
                                self.theano_graph.compute_weights(),
                                # mode=NanGuardMode(nan_is_error=True),
                                on_unused_input='warn',
                                allow_input_downcast=False,
                                profile=False)
        if inplace is True:
            self.theano_function = th_fn

        if debug is True:
            print('Level of Optimization: ', theano.config.optimizer)
            print('Device: ', theano.config.device)
            print('Precision: ', self.dtype)
            print('Number of faults: ', self.additional_data.structure_data.df.loc['values', 'number faults'])
        print('Compilation Done!')
        return th_fn


class InterpolatorScalar(Interpolator):

    def __init__(self, surface_points: "SurfacePoints", orientations: "Orientations", grid: "Grid",
                 surfaces: "Surfaces", series, faults: "Faults", additional_data: "AdditionalData", **kwargs):

        super(InterpolatorScalar, self).__init__(surface_points, orientations, grid, surfaces, series, faults,
                                                 additional_data, **kwargs)

    def get_python_input_zx(self, fault_drift=None):
        """
             Get values from the data objects used during the interpolation:
                 - dip positions XYZ
                 - dip angles
                 - azimuth
                 - polarity
                 - surface_points coordinates XYZ
             Returns:
                 (list)
             """
        # orientations, this ones I tile them inside theano. PYTHON VAR
        dips_position = self.orientations.df[['X_r', 'Y_r', 'Z_r']].values
        dip_angles = self.orientations.df["dip"].values
        azimuth = self.orientations.df["azimuth"].values
        polarity = self.orientations.df["polarity"].values
        surface_points_coord = self.surface_points.df[['X_r', 'Y_r', 'Z_r']].values
        grid = self.grid.values_r

        if fault_drift is None:
            fault_drift = np.zeros((0, grid.shape[0] + 2 * self.len_series_i.sum()))

      #      fault_drift = np.zeros((0, grid.shape[0] + surface_points_coord.shape[0]))

        # Set all in a list casting them in the chosen dtype
        idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity, surface_points_coord,
                                                  fault_drift, grid)]
        return idl

    def compile_th_fn(self, weights=None, grid=None, inplace=False, debug=False):
        """

        Args:
            weights: Constant weights
            grid:  Constant grids
            inplace:
            debug:

        Returns:

        """
        self.set_theano_shared_kriging()
        self.set_theano_shared_structure_surfaces()
        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.theano_graph.input_parameters_kriging_export
        print('Compiling theano function...')

        if weights is None:
            weights = self.theano_graph.compute_weights()
        else:
            weights = theano.shared(weights)

        if grid is None:
            grid = self.theano_graph.grid_val_T
        else:
            grid = theano.shared(grid)

        th_fn = theano.function(input_data_T,
                                self.theano_graph.compute_scalar_field(weights, grid),
                                # mode=NanGuardMode(nan_is_error=True),
                                on_unused_input='ignore',
                                allow_input_downcast=False,
                                profile=False)

        if inplace is True:
            self.theano_function = th_fn

        if debug is True:
            print('Level of Optimization: ', theano.config.optimizer)
            print('Device: ', theano.config.device)
            print('Precision: ', self.dtype)
            print('Number of faults: ', self.additional_data.structure_data.df.loc['values', 'number faults'])
        print('Compilation Done!')
        return th_fn


class InterpolatorBlock(Interpolator):

    def __init__(self, surface_points: "SurfacePoints", orientations: "Orientations", grid: "Grid",
                 surfaces: "Surfaces", series: Series, faults: "Faults", additional_data: "AdditionalData", **kwargs):

        super(InterpolatorBlock, self).__init__(surface_points, orientations, grid, surfaces, series,
                                                faults, additional_data, **kwargs)
        self.theano_function_formation = None
        self.theano_function_faults = None

    def get_python_input_block(self, fault_drift=None):
        """
             Get values from the data objects used during the interpolation:
                 - dip positions XYZ
                 - dip angles
                 - azimuth
                 - polarity
                 - surface_points coordinates XYZ
             Returns:
                 (list)
             """
        # orientations, this ones I tile them inside theano. PYTHON VAR
        dips_position = self.orientations.df[['X_r', 'Y_r', 'Z_r']].values
        dip_angles = self.orientations.df["dip"].values
        azimuth = self.orientations.df["azimuth"].values
        polarity = self.orientations.df["polarity"].values
        surface_points_coord = self.surface_points.df[['X_r', 'Y_r', 'Z_r']].values
        grid = self.grid.values_r
        if fault_drift is None:
            fault_drift = np.zeros((0, grid.shape[0] + 2 * self.len_series_i.sum()))

         #   fault_drift = np.zeros((0, grid.shape[0] + surface_points_coord.shape[0]))

        values_properties = self.surfaces.df.iloc[:, self.surfaces._n_properties:].values.astype(self.dtype).T

        # Set all in a list casting them in the chosen dtype
        idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity, surface_points_coord,
                                                  fault_drift, grid, values_properties)]
        return idl

    def compile_th_fn_formation_block(self, Z_x=None, weights=None, grid=None, values_properties=None, inplace=False,
                                  debug=False):
        """

        Args:
            weights: Constant weights
            grid:  Constant grids
            inplace:
            debug:

        Returns:

        """
        self.set_theano_shared_kriging()
        self.set_theano_shared_structure_surfaces()
        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.theano_graph.input_parameters_block
        print('Compiling theano function...')

        if weights is None:
            weights = self.theano_graph.compute_weights()
        else:
            weights = theano.shared(weights)

        if grid is None:
            grid = self.theano_graph.grid_val_T
        else:
            grid = theano.shared(grid)

        if values_properties is None:
            values_properties = self.theano_graph.values_properties_op
        else:
            values_properties = theano.shared(values_properties)

        if Z_x is None:
            Z_x = self.theano_graph.compute_scalar_field(weights, grid)
        else:
            Z_x = theano.shared(Z_x)

        th_fn = theano.function(input_data_T,
                                self.theano_graph.compute_formation_block(
                                    Z_x,
                                    self.theano_graph.get_scalar_field_at_surface_points(Z_x),
                                    values_properties
                                ),
                                # mode=NanGuardMode(nan_is_error=True),
                                on_unused_input='ignore',
                                allow_input_downcast=False,
                                profile=False)

        if inplace is True:
            self.theano_function_formation = th_fn

        if debug is True:
            print('Level of Optimization: ', theano.config.optimizer)
            print('Device: ', theano.config.device)
            print('Precision: ', self.dtype)
            print('Number of faults: ', self.additional_data.structure_data.df.loc['values', 'number faults'])
        print('Compilation Done!')

        return th_fn

    def compile_th_fn_fault_block(self, Z_x=None, weights=None, grid=None, values_properties=None,
                                  inplace=False, debug=False):
        """

        Args:
            weights: Constant weights
            grid:  Constant grids
            inplace:
            debug:

        Returns:

        """
        self.set_theano_shared_kriging()
        self.set_theano_shared_structure_surfaces()
        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.theano_graph.input_parameters_block
        print('Compiling theano function...')

        if weights is None:
            weights = self.theano_graph.compute_weights()
        else:
            weights = theano.shared(weights)

        if grid is None:
            grid = self.theano_graph.grid_val_T
        else:
            grid = theano.shared(grid)

        if values_properties is None:
            values_properties = self.theano_graph.values_properties_op
        else:
            values_properties = theano.shared(values_properties)

        if Z_x is None:
            Z_x = self.theano_graph.compute_scalar_field(weights, grid)
        else:
            Z_x = theano.shared(Z_x)

        th_fn = theano.function(input_data_T,
                                self.theano_graph.compute_fault_block(
                                    Z_x,
                                    self.theano_graph.get_scalar_field_at_surface_points(Z_x),
                                    values_properties,
                                    0,
                                    grid
                                 ),
                                # mode=NanGuardMode(nan_is_error=True),
                                on_unused_input='ignore',
                                allow_input_downcast=False,
                                profile=False)

        if inplace is True:
            self.theano_function_faults = th_fn

        if debug is True:
            print('Level of Optimization: ', theano.config.optimizer)
            print('Device: ', theano.config.device)
            print('Precision: ', self.dtype)
            print('Number of faults: ', self.additional_data.structure_data.df.loc['values', 'number faults'])
        print('Compilation Done!')
        return th_fn


@setdoc_pro(ds.ctrl)
@setdoc([Interpolator.__doc__])
class InterpolatorModel(Interpolator):
    """
    Child class of :class:`Interpolator` which set the shared variables and compiles the theano
    graph to compute the geological model, i.e. lithologies.

    Attributes:
        compute_weights_ctrl (list[bool]): [s0]
        compute_scalar_ctrl (list[bool]):
        compute_block_ctrl (list[bool]):

    Interpolator Doc
    """
    def __init__(self, surface_points: "SurfacePoints", orientations: "Orientations", grid: "Grid",
                 surfaces: "Surfaces", series, faults: "Faults", additional_data: "AdditionalData", **kwargs):

        super().__init__(surface_points, orientations, grid, surfaces, series, faults,
                         additional_data, **kwargs)
        self.len_series_i = np.zeros(1)
        self.len_series_o = np.zeros(1)
        self.len_series_u = np.zeros(1)
        self.len_series_f = np.zeros(1)
        self.len_series_w = np.zeros(1)

        self.set_initial_results()

        n_series = 1000

        self.compute_weights_ctrl = np.ones(n_series, dtype=bool)
        self.compute_scalar_ctrl = np.ones(n_series, dtype=bool)
        self.compute_block_ctrl = np.ones(n_series, dtype=bool)

    def reset_flow_control_initial_results(self, reset_weights=True, reset_scalar=True, reset_block=True):
        """
        Method to reset to the initial state all the recompute ctrl. After calling this method next time
         gp.compute_model is called, everything will be computed. Panic bottom.

        Args:
            reset_weights (bool):
            reset_scalar (bool):
            reset_block (bool):

        Returns:
            True
        """
        n_series = self.additional_data.get_additional_data()['values']['Structure', 'number series']
        x_to_interp_shape = self.grid.values_r.shape[0] + 2 * self.len_series_i.sum()

        if reset_weights is True:
            self.compute_weights_ctrl = np.ones(1000, dtype=bool)
            self.theano_graph.weights_vector.set_value(np.zeros((self.len_series_w.sum())))

        if reset_scalar is True:
            self.compute_scalar_ctrl = np.ones(1000, dtype=bool)
            self.theano_graph.scalar_fields_matrix.set_value(
                np.zeros((n_series, x_to_interp_shape), dtype=self.dtype))

        if reset_block is True:
            self.compute_block_ctrl = np.ones(1000, dtype=bool)
            self.theano_graph.mask_matrix.set_value(np.zeros((n_series, x_to_interp_shape), dtype='bool'))
            self.theano_graph.block_matrix.set_value(
                np.zeros((n_series, self.surfaces.df.iloc[:, self.surfaces._n_properties:].values.shape[1],
                          x_to_interp_shape), dtype=self.dtype))
        return True

    def set_flow_control(self):
        """
        Initialize the ctrl vectors to the number of series size.

        Returns:
            True
        """
        n_series = 1000
        self.compute_weights_ctrl = np.ones(n_series, dtype=bool)
        self.compute_scalar_ctrl = np.ones(n_series, dtype=bool)
        self.compute_block_ctrl = np.ones(n_series, dtype=bool)
        return True

    @setdoc_pro(reset_flow_control_initial_results.__doc__)
    def set_all_shared_parameters(self, reset_ctrl=False):
        """
        Set all theano shared parameters required for the computation of lithology

        Args:
            reset_ctrl (bool): If true, [s0]

        Returns:
            True
        """
        self.set_theano_shared_loop()
        self.set_theano_shared_relations()
        self.set_theano_shared_kriging()
        self.set_theano_shared_structure_surfaces()

        if reset_ctrl is True:
            self.reset_flow_control_initial_results()

        return True

    @setdoc_pro(reset_flow_control_initial_results.__doc__)
    def set_theano_shared_structure(self, reset_ctrl=False):
        """
        Set all theano shared variable dependent on :class:`Structure`.

        Args:
            reset_ctrl (bool): If true, [s0]

        Returns:
            True

        """
        self.set_theano_shared_loop()
        self.set_theano_shared_relations()
        self.set_theano_shared_structure_surfaces()
        # universal grades
        self.theano_graph.n_universal_eq_T.set_value(
            list(self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype('int32')))

        if reset_ctrl is True:
            self.reset_flow_control_initial_results()
        return True

    def _compute_len_series(self):
        self.len_series_i = self.additional_data.structure_data.df.loc['values', 'len series surface_points'] - \
                            self.additional_data.structure_data.df.loc['values', 'number surfaces per series']
        if self.len_series_i.shape[0] == 0:
            self.len_series_i = np.zeros(1, dtype=int)

        self.len_series_o = self.additional_data.structure_data.df.loc['values', 'len series orientations'].astype(
            'int32')
        if self.len_series_o.shape[0] == 0:
            self.len_series_o = np.zeros(1, dtype=int)

        self.len_series_u = self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype('int32')
        if self.len_series_u.shape[0] == 0:
            self.len_series_u = np.zeros(1, dtype=int)

        self.len_series_f = self.faults.faults_relations_df.sum(axis=0).values.astype('int32')[
                            :self.additional_data.get_additional_data()['values']['Structure', 'number series']]
        if self.len_series_f.shape[0] == 0:
            self.len_series_f = np.zeros(1, dtype=int)

        self.len_series_w = self.len_series_i + self.len_series_o * 3 + self.len_series_u + self.len_series_f

    def set_theano_shared_loop(self):
        """Set the theano shared variables that are looped for each series."""
        self._compute_len_series()

        self.theano_graph.len_series_i.set_value(np.insert(self.len_series_i.cumsum(), 0, 0).astype('int32'))
        self.theano_graph.len_series_o.set_value(np.insert(self.len_series_o.cumsum(), 0, 0).astype('int32'))
        self.theano_graph.len_series_w.set_value(np.insert(self.len_series_w.cumsum(), 0, 0).astype('int32'))

        # Number of surfaces per series. The function is not pretty but the result is quite clear
        n_surfaces_per_serie = np.insert(
            self.additional_data.structure_data.df.loc['values', 'number surfaces per series'].cumsum(), 0, 0). \
            astype('int32')
        self.theano_graph.n_surfaces_per_series.set_value(n_surfaces_per_serie)
        self.theano_graph.n_universal_eq_T.set_value(
            list(self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype('int32')))

    @setdoc_pro(set_theano_shared_loop.__doc__)
    def set_theano_shared_weights(self):
        """Set the theano shared weights and [s0]"""
        self.set_theano_shared_loop()
        self.theano_graph.weights_vector.set_value(np.zeros((self.len_series_w.sum())))

    def set_theano_shared_fault_relation(self):
        """Set the theano shared variable with the fault relation"""
        self.theano_graph.fault_relation.set_value(self.faults.faults_relations_df.values)

    def set_theano_shared_is_fault(self):
        """Set theano shared variable which controls if a series is fault or not"""
        self.theano_graph.is_fault.set_value(self.faults.df['isFault'].values)

    def set_theano_shared_is_finite(self):
        """Set theano shared variable which controls if a fault is finite or not"""
        self.theano_graph.is_finite_ctrl.set_value(self.faults.df['isFinite'].values)

    def set_theano_shared_onlap_erode(self):
        """Set the theano variables which control the masking patterns according to the uncomformity relation"""
        n_series = self.additional_data.structure_data.df.loc['values', 'number series']

        is_erosion = self.series.df['BottomRelation'].values[:n_series] == 'Erosion'
        is_onlap = np.roll(self.series.df['BottomRelation'].values[:n_series] == 'Onlap', 1)

        is_erosion[-1] = False

        # this comes from the series df
        self.theano_graph.is_erosion.set_value(is_erosion)
        self.theano_graph.is_onlap.set_value(is_onlap)

    def set_theano_shared_faults(self):
        """Set all theano shared variables wich controls the faults behaviour"""

        self.set_theano_shared_fault_relation()
        # This comes from the faults df
        self.set_theano_shared_is_fault()
        self.set_theano_shared_is_finite()

    def set_theano_shared_relations(self):
        """Set all theano shared variables that control all the series interactions with each other"""
        self.set_theano_shared_fault_relation()
        # This comes from the faults df
        self.set_theano_shared_is_fault()
        self.set_theano_shared_is_finite()
        self.set_theano_shared_onlap_erode()

    def set_initial_results(self):
        """
        Initialize all the theano shared variables where we store the final results of the interpolation.
        This function must be called always after set_theano_shared_loop

        Returns:
            True
        """
        self._compute_len_series()

        x_to_interp_shape = self.grid.values_r.shape[0] + 2 * self.len_series_i.sum()
        n_series = self.additional_data.structure_data.df.loc['values', 'number series']

        self.theano_graph.weights_vector.set_value(np.zeros((self.len_series_w.sum())))
        self.theano_graph.scalar_fields_matrix.set_value(
            np.zeros((n_series, x_to_interp_shape), dtype=self.dtype))

        self.theano_graph.mask_matrix.set_value(np.zeros((n_series, x_to_interp_shape), dtype='bool'))
        self.theano_graph.block_matrix.set_value(
            np.zeros((n_series, self.surfaces.df.iloc[:, self.surfaces._n_properties:].values.shape[1],
                      x_to_interp_shape), dtype=self.dtype))
        return True

    def set_initial_results_matrices(self):
        """
        Initialize all the theano shared variables where we store the final results of the interpolation except the
        kriging weights vector.


        Returns:
            True
        """
        self._compute_len_series()

        x_to_interp_shape = self.grid.values_r.shape[0] + 2 * self.len_series_i.sum()
        n_series = self.additional_data.structure_data.df.loc['values', 'number series']

        self.theano_graph.scalar_fields_matrix.set_value(
            np.zeros((n_series, x_to_interp_shape), dtype=self.dtype))

        self.theano_graph.mask_matrix.set_value(np.zeros((n_series, x_to_interp_shape), dtype='bool'))
        self.theano_graph.block_matrix.set_value(
            np.zeros((n_series, self.surfaces.df.iloc[:, self.surfaces._n_properties:].values.shape[1],
                      x_to_interp_shape), dtype=self.dtype))

    def modify_results_matrices_pro(self):
        """Modify all theano shared matrices to the right size according to the structure data. This method allows
        to change the size of the results without having the recompute all series"""
        old_len_i = self.len_series_i
        new_len_i = self.additional_data.structure_data.df.loc['values', 'len series surface_points'] - \
            self.additional_data.structure_data.df.loc['values', 'number surfaces per series']
        if new_len_i.shape[0] != old_len_i[0]:
            self.set_initial_results()
        else:
            scalar_fields_matrix = self.theano_graph.scalar_fields_matrix.get_value()
            mask_matrix = self.theano_graph.mask_matrix.get_value()
            block_matrix = self.theano_graph.block_matrix.get_value()

            len_i_diff = new_len_i - old_len_i
            for e, i in enumerate(len_i_diff):
                loc = self.grid.values_r.shape[0] + old_len_i[e]
                i *= 2
                if i == 0:
                    pass
                elif i > 0:
                    self.theano_graph.scalar_fields_matrix.set_value(
                        np.insert(scalar_fields_matrix, [loc], np.zeros(i), axis=1))
                    self.theano_graph.mask_matrix.set_value(np.insert(mask_matrix, [loc], np.zeros(i), axis=1))
                    self.theano_graph.block_matrix.set_value(np.insert(block_matrix, [loc], np.zeros(i), axis=2))

                else:
                    self.theano_graph.scalar_fields_matrix.set_value(
                        np.delete(scalar_fields_matrix, np.arange(loc, loc+i, -1) - 1, axis=1))
                    self.theano_graph.mask_matrix.set_value(
                        np.delete(mask_matrix, np.arange(loc, loc+i, -1) - 1, axis=1))
                    self.theano_graph.block_matrix.set_value(
                        np.delete(block_matrix, np.arange(loc, loc+i, -1) - 1, axis=2))

        self.modify_results_weights()

    def modify_results_weights(self):
        """Modify the theano shared weights vector according to the structure.
        """
        old_len_w = self.len_series_w
        self._compute_len_series()
        new_len_w = self.len_series_w
        if new_len_w.shape[0] != old_len_w[0]:
            self.set_initial_results()
        else:
            weights = self.theano_graph.weights_vector.get_value()
            len_w_diff = new_len_w - old_len_w
            for e, i in enumerate(len_w_diff):
                print(len_w_diff, weights)
                if i == 0:
                    pass
                elif i > 0:
                    self.theano_graph.weights_vector.set_value(np.insert(weights, old_len_w[e], np.zeros(i)))
                else:
                    print(np.delete(weights, np.arange(old_len_w[e],  old_len_w[e] + i, -1)-1))
                    self.theano_graph.weights_vector.set_value(
                        np.delete(weights, np.arange(old_len_w[e],  old_len_w[e] + i, -1)-1))

    def get_python_input_block(self, append_control=True, fault_drift=None):
        """
        Get values from the data objects used during the interpolation:
             - dip positions XYZ
             - dip angles
             - azimuth
             - polarity
             - surface_points coordinates XYZ

        Args:
            append_control (bool): If true append the ctrl vectors to the input list
            fault_drift (Optional[np.array]): matrix with per computed faults to drift the model

        Returns:
            list: list of arrays with all the input parameters to the theano function
        """
        # orientations, this ones I tile them inside theano. PYTHON VAR
        dips_position = self.orientations.df[['X_r', 'Y_r', 'Z_r']].values
        dip_angles = self.orientations.df["dip"].values
        azimuth = self.orientations.df["azimuth"].values
        polarity = self.orientations.df["polarity"].values
        surface_points_coord = self.surface_points.df[['X_r', 'Y_r', 'Z_r']].values
        grid = self.grid.values_r
        if fault_drift is None:
            fault_drift = np.zeros((0, grid.shape[0] + 2 * self.len_series_i.sum()))

        values_properties = self.surfaces.df.iloc[:, self.surfaces._n_properties:].values.astype(self.dtype).T

        # Set all in a list casting them in the chosen dtype
        idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity, surface_points_coord,
                                                  fault_drift, grid, values_properties)]
        if append_control is True:
            idl.append(self.compute_weights_ctrl)
            idl.append(self.compute_scalar_ctrl)
            idl.append(self.compute_block_ctrl)

        return idl

    def print_theano_shared(self):
        """Print many of the theano shared variables"""
        print('len sereies i', self.theano_graph.len_series_i.get_value())
        print('len sereies o', self.theano_graph.len_series_o.get_value())
        print('len sereies w', self.theano_graph.len_series_w.get_value())
        print('n surfaces per series', self.theano_graph.n_surfaces_per_series.get_value())
        print('n universal eq',self.theano_graph.n_universal_eq_T.get_value())
        print('is finite', self.theano_graph.is_finite_ctrl.get_value())
        print('is erosion', self.theano_graph.is_erosion.get_value())
        print('is onlap', self.theano_graph.is_onlap.get_value())

    def compile_th_fn(self, inplace=False,
                      debug=False):
        """
        Compile and create the theano function which can be evaluated to compute the geological models

        Args:

            inplace (bool): If true add the attribute theano.function to the object inplace
            debug (bool): If true print some of the theano flags

        Returns:
            theano.function: function that computes the whole interpolation
        """
        from theano.compile.nanguardmode import NanGuardMode

        self.set_all_shared_parameters(reset_ctrl=False)
        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.theano_graph.input_parameters_loop
        print('Compiling theano function...')

        th_fn = theano.function(input_data_T,
                                self.theano_graph.compute_series(),
                                updates=[(self.theano_graph.block_matrix, self.theano_graph.new_block),
                                         (self.theano_graph.weights_vector, self.theano_graph.new_weights),
                                         (self.theano_graph.scalar_fields_matrix, self.theano_graph.new_scalar),
                                         (self.theano_graph.mask_matrix, self.theano_graph.new_mask)],
                             #   mode=NanGuardMode(nan_is_error=True),
                                on_unused_input='ignore',
                                allow_input_downcast=False,
                                profile=False)

        if inplace is True:
            self.theano_function = th_fn

        if debug is True:
            print('Level of Optimization: ', theano.config.optimizer)
            print('Device: ', theano.config.device)
            print('Precision: ', self.dtype)
            print('Number of faults: ', self.additional_data.structure_data.df.loc['values', 'number faults'])
        print('Compilation Done!')

        return th_fn


@setdoc([InterpolatorModel.__doc__], indent=False)
class InterpolatorGravity(InterpolatorModel):
    """
    Child class of :class:`InterpolatorModel` which set the specific shared variables for the gravity computation and
    compiles the theano graph to compute the geological model, i.e. lithologies and the forward gravity.

    InterpolatorModel Doc

    """

    def set_theano_shared_tz_kernel(self, tz=None):
        """Set the theano component tz to each voxel"""

        if tz is None:
            try:
                tz = self.calculate_tz()
            except AttributeError:
                raise AttributeError('You need to calculate or pass tz first.')
        self.theano_graph.tz.set_value(tz.astype(self.dtype))

    def compile_th_fn(self, density=None, pos_density=None, inplace=False,
                      debug=False):
        """
        Compile and create the theano function which can be evaluated to compute the forward gravity response for
        a given kernel.

        Args:
            density (Optional[np.array]): array of the same size as the grid.values with the correspondant value of
             density per voxel.
            pos_density (Optional[int]): if density is not passed, pos_density will define which values (i.e. column
             after id in the :class:`Surface` df is the density)
            inplace (bool): If true add the attribute theano.function to the object inplace
            debug (bool): If true print some of the theano flags

        Returns:
            theano.function: function that computes the whole interpolation
        """
        assert density is not None or pos_density is not None, 'If you do not pass the density block you need to pass' \
                                                               'the position of surface values where density is' \
                                                               ' assigned'

        from theano.compile.nanguardmode import NanGuardMode

        self.set_all_shared_parameters(reset_ctrl=False)
        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.theano_graph.input_parameters_loop
        print('Compiling theano function...')
        if density is None:
            assert pos_density is not None, 'If a density block is not passed, you need to specify which interpolated' \
                                            'value is density. See :class:`Surface`'
            density = self.theano_graph.compute_series()[0][pos_density, :- 2 * self.theano_graph.len_points]

        else:
            density = theano.shared(density)

        th_fn = theano.function(input_data_T,
                                self.theano_graph.compute_forward_gravity(density),
                                updates=[(self.theano_graph.block_matrix, self.theano_graph.new_block),
                                         (self.theano_graph.weights_vector, self.theano_graph.new_weights),
                                         (self.theano_graph.scalar_fields_matrix, self.theano_graph.new_scalar),
                                         (self.theano_graph.mask_matrix, self.theano_graph.new_mask)],
                             #   mode=NanGuardMode(nan_is_error=True),
                                on_unused_input='ignore',
                                allow_input_downcast=False,
                                profile=False)

        if inplace is True:
            self.theano_function = th_fn

        if debug is True:
            print('Level of Optimization: ', theano.config.optimizer)
            print('Device: ', theano.config.device)
            print('Precision: ', self.dtype)
            print('Number of faults: ', self.additional_data.structure_data.df.loc['values', 'number faults'])
        print('Compilation Done!')

        return th_fn

    def calculate_tz(self):
        from gempy.assets.geophysics import GeophysicsPreprocessing
        g = GeophysicsPreprocessing(self.grid.centered_grid)

        return g.set_tz_kernel()
