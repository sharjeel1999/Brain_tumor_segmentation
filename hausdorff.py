import numpy
from scipy.ndimage import _ni_support
from scipy.ndimage import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from scipy.ndimage import label, find_objects
from scipy.stats import pearsonr


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = numpy.asarray(voxelspacing, dtype=numpy.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == numpy.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == numpy.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds


def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.
    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.
    See also
    --------
    :func:`hd`
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = numpy.percentile(numpy.hstack((hd1, hd2)), 95)
    return hd95









# import numpy as np
# import numba
# import hausdorff.distances as distances
# from inspect import getmembers

# def _find_available_functions(module_name):
# 	all_members = getmembers(module_name)
# 	available_functions = [member[0] for member in all_members 
# 						   if isinstance(member[1], numba.core.registry.CPUDispatcher)]
# 	return available_functions

# @numba.jit(nopython=True, fastmath=True)
# def _hausdorff(XA, XB, distance_function):
# 	nA = XA.shape[0]
# 	nB = XB.shape[0]
# 	cmax = 0.
# 	for i in range(nA):
# 		cmin = np.inf
# 		for j in range(nB):
# 			d = distance_function(XA[i,:], XB[j,:])
# 			if d<cmin:
# 				cmin = d
# 			if cmin<cmax:
# 				break
# 		if cmin>cmax and np.inf>cmin:
# 			cmax = cmin
# 	for j in range(nB):
# 		cmin = np.inf
# 		for i in range(nA):
# 			d = distance_function(XA[i,:], XB[j,:])
# 			if d<cmin:
# 				cmin = d
# 			if cmin<cmax:
# 				break
# 		if cmin>cmax and np.inf>cmin:
# 			cmax = cmin
# 	return cmax

# def hausdorff_distance(XA, XB, distance='euclidean'):
# 	assert isinstance(XA, np.ndarray) and isinstance(XB, np.ndarray), \
# 		'arrays must be of type numpy.ndarray'
# 	assert np.issubdtype(XA.dtype, np.number) and np.issubdtype(XA.dtype, np.number), \
# 		'the arrays data type must be numeric'
# 	assert XA.ndim == 2 and XB.ndim == 2, \
# 		'arrays must be 2-dimensional'
# 	assert XA.shape[1] == XB.shape[1], \
# 		'arrays must have equal number of columns'
# 	
# 	if isinstance(distance, str):
# 		assert distance in _find_available_functions(distances), \
# 			'distance is not an implemented function'
# 		if distance == 'haversine':
# 			assert XA.shape[1] >= 2, 'haversine distance requires at least 2 coordinates per point (lat, lng)'
# 			assert XB.shape[1] >= 2, 'haversine distance requires at least 2 coordinates per point (lat, lng)'
# 		distance_function = getattr(distances, distance)
# 	elif callable(distance):
# 		distance_function = distance
# 	else:
# 		raise ValueError("Invalid input value for 'distance' parameter.")
# 	return _hausdorff(XA, XB, distance_function)