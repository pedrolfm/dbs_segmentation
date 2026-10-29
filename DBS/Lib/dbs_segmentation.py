import slicer, vtk
import numpy as np
from scipy.ndimage import label as cc_label, find_objects as cc_find_objects, binary_closing
from ModelRegistration import ModelRegistrationLogic

def _get_or_create_segmentation(volumeNode, baseName="Segmentation"):
    """Return a segmentation node named `baseName` if present; otherwise create it.
    Always ensures reference geometry and display nodes exist."""
    try:
        segNode = slicer.util.getNode(baseName)
        if segNode.GetClassName() != "vtkMRMLSegmentationNode":
            # If a different node has that name, make a new proper one
            raise slicer.util.MRMLNodeNotFoundException("")
    except slicer.util.MRMLNodeNotFoundException:
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", baseName)
    segNode.CreateDefaultDisplayNodes()
    segNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
    return segNode

def _ensure_target_segment(segmentationNode, preferredName="tube"):
    """Return a segmentId to work with:
       - if there are segments, use the last one (latest added)
       - otherwise create an empty segment (named `preferredName`) and use it
    """
    seg = segmentationNode.GetSegmentation()
    n = seg.GetNumberOfSegments()
    if n == 0:
        # Create one and grab its id
        seg.AddEmptySegment(preferredName)
        segmentId = seg.GetNthSegmentID(seg.GetNumberOfSegments() - 1)
    else:
        # Use the most recently added segment
        segmentId = seg.GetNthSegmentID(n - 1)
    return segmentId

def _numpy_from_segment(segmentationNode, segmentId, referenceVolumeNode):
    """Export the given segment to a temporary LabelMapVolume and return a NumPy mask [Z,Y,X]."""
    # Make sure binary labelmap representation exists
    binaryName = slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()
    segmentationNode.GetSegmentation().CreateRepresentation(binaryName)

    # Export to a labelmap aligned with the volume geometry
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        segmentationNode, [segmentId], labelmapNode, referenceVolumeNode
    )
    arr = slicer.util.arrayFromVolume(labelmapNode)  # [Z,Y,X]
    slicer.mrmlScene.RemoveNode(labelmapNode)
    if arr is None:
        # Guard against empty/failed export
        arr = np.zeros(slicer.util.arrayFromVolume(referenceVolumeNode).shape, dtype=np.uint8)
    return (arr > 0).astype(np.uint8)

def fit_circle(x, y):
    # algebraic least-squares circle fit
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = c[0] / 2, c[1] / 2
    r = np.sqrt(c[2] + cx**2 + cy**2)
    return cx, cy, r

def checkCenter(segNode,volNode):
    import numpy as np
    try:
        from scipy import ndimage
    except ImportError:
        slicer.util.pip_install('scipy'); from scipy import ndimage

    segId = segNode.GetSegmentation().GetSegmentIdBySegmentName('Segment_1')

    arr = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segId, volNode)  # indexed [k, j, i]
    m = vtk.vtkMatrix4x4(); volNode.GetIJKToRASMatrix(m)
    ijkToRas = slicer.util.arrayFromVTKMatrix(m)

    centers = []
    for k in range(arr.shape[0]):
        sl = arr[k] > 0
        if sl.sum() < 20:
            continue
        edge = sl & ~ndimage.binary_erosion(sl)   # ring boundary pixels only
        j, i = np.nonzero(edge)
        cx, cy, r = fit_circle(i.astype(float), j.astype(float))
        centers.append((ijkToRas @ [cx, cy, k, 1])[:3])
    centers = np.array(centers)
    print(centers)


    # Smooth: fit R and A as polynomials of S (removes slice-to-slice jitter)
    s = centers[:, 2]
    fitR = np.polyval(np.polyfit(s, centers[:, 0], 3), s)
    fitA = np.polyval(np.polyfit(s, centers[:, 1], 3), s)
    smooth = np.column_stack([fitR, fitA, s])

    curve = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode', 'Centerline')
    for p in smooth[::5]:                        # every 5th point as a control point
        curve.AddControlPoint(p)
    print('Centerline length (mm):', curve.GetCurveLengthWorld())

def point_to_line(p, a, b, segment=False):
    """Shortest distance from point p to the line through a and b.
    segment=True limits it to the segment between a and b."""
    p, a, b = map(np.asarray, (p, a, b))
    u = (b - a) / np.linalg.norm(b - a)     # unit direction of the line
    t = np.dot(p - a, u)                     # signed position along the line (mm from a)
    if segment:
        t = np.clip(t, 0.0, np.linalg.norm(b - a))
    closest = a + t * u
    return np.linalg.norm(p - closest), closest, t


def run_seg_pipeline(path_or_volumeNode, target, stl_path):
    # ---------------------------
    # INPUT HANDLING
    # ---------------------------
    if isinstance(path_or_volumeNode, str):
        volumeNode = slicer.util.loadVolume(path_or_volumeNode)
        if volumeNode is None:
            raise RuntimeError(f"Failed to load volume from: {path_or_volumeNode}")
    else:
        volumeNode = path_or_volumeNode
        if volumeNode is None:
            raise ValueError("No input volume node provided.")

    # ============================================================
    # ================ PART 1 (threshold segmentation) ===========
    # ============================================================
    segmentationNode = _get_or_create_segmentation(volumeNode, baseName="Segmentation")
    segmentId = _ensure_target_segment(segmentationNode, preferredName="tube")

    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setSourceVolumeNode(volumeNode)
    segmentEditorWidget.setCurrentSegmentID(segmentId)

    segmentEditorWidget.setActiveEffectByName("Threshold")
    effect = segmentEditorWidget.activeEffect()

    ###### testing automatic thresholding ######
    # Convert image data to numpy array
    imageData = slicer.util.arrayFromVolume(volumeNode)
    flattened = imageData.flatten()

    # Compute percentile-based thresholds
    lower_percentile = np.percentile(flattened, 99.0)
    upper_percentile = np.percentile(flattened, 99.9)

    print(f"Auto-threshold range: {lower_percentile:.2f} to {upper_percentile:.2f}")
    effect.setParameter("MinimumThreshold", lower_percentile)
    effect.setParameter("MaximumThreshold", upper_percentile)

    effect.self().onApply()

    segmentEditorWidget = None
    slicer.mrmlScene.RemoveNode(segmentEditorNode)

    segmentationNode.CreateClosedSurfaceRepresentation()

    # ============================================================
    # ============ PART 2 (remove unnecessary parts) =============
    # ============================================================

    #seg_np = _numpy_from_segment(segmentationNode, segmentId, volumeNode)

#    labeled_array, _ = cc_label(seg_np)
#    object_slices = cc_find_objects(labeled_array)

#    ASPECT_RATIO_THRESHOLD = 4
#    tube_labels = []
#    for i, object_slice in enumerate(object_slices):
#        if object_slice is None:
#            continue
#        z_slice, y_slice, x_slice = object_slice
#        height = y_slice.stop - y_slice.start
#        width  = x_slice.stop - x_slice.start
#        aspect_ratio = (height / width) if width > 0 else height * 10
#        if aspect_ratio > ASPECT_RATIO_THRESHOLD:
#            tube_labels.append(i + 1)

#    final_tube_mask = np.isin(labeled_array, tube_labels).astype(np.uint8)
#
 #   slicer.util.updateSegmentBinaryLabelmapFromArray(
 #       final_tube_mask, segmentationNode, segmentId, volumeNode
 #   )
#    segmentationNode.CreateClosedSurfaceRepresentation()

    # ============================================================
    # =============== PART 3 (smoothing + bbox fill)  ============
    # ============================================================
    ## 09/18/2025 testing
    
    # bounding boxes (sagittal + coronal)
  #  final_mask = np.zeros_like(final_tube_mask, dtype=bool)
  #
  #  # Z-planes
  #  y_range_z = float('-inf')
  #  y_range_slice = slice(None)
  #  for z in range(final_tube_mask.shape[0]):
  #      current_slice = final_tube_mask[z, :, :]
  #      coords = np.argwhere(current_slice)
  #      if coords.size:
  #          y_min, x_min = coords.min(axis=0)
  #          y_max, x_max = coords.max(axis=0)
  #          y_range_current = y_max - y_min
  #          if y_range_current > y_range_z:
  #              y_range_z = y_range_current
  #             y_range_slice = slice(y_min, y_max + 1)
  #          final_mask[z, y_min:y_max+1, x_min:x_max+1] = True
    
    # 09/23/2025
  #  for z in range(final_tube_mask.shape[0]):
  #      current_slice = final_tube_mask[z, :, :]
  #      coords = np.argwhere(current_slice)
  #      if coords.size:
  #          y_min, x_min = coords.min(axis=0)
  #          y_max, x_max = coords.max(axis=0)
  #          final_mask[z, y_range_slice, x_min:x_max+1] = True

    # X-columns
  #  y_range_x = float('-inf')
  #  y_range_slice = slice(None)
  #  for x in range(final_tube_mask.shape[2]):
  #      current_slice = final_tube_mask[:, :, x]
  #      coords = np.argwhere(current_slice)
  #      if coords.size:
  #          z_min, y_min = coords.min(axis=0)
  #          z_max, y_max = coords.max(axis=0)
  #          y_range_current = y_max - y_min
  #         if y_range_current > y_range_x:
  #             y_range_x = y_range_current
  #             y_range_slice = slice(y_min, y_max + 1)
  #         final_mask[z_min:z_max+1, y_min:y_max+1, x] = True

  # for x in range(final_tube_mask.shape[2]):
  #     current_slice = final_tube_mask[:, :, x]
  #     coords = np.argwhere(current_slice)
  #     if coords.size:
  #         z_min, y_min = coords.min(axis=0)
  #         z_max, y_max = coords.max(axis=0)
  #         final_mask[z_min:z_max+1, y_range_slice, x] = True

  # print("Updating the segment with the bounding box result...")
  # slicer.util.updateSegmentBinaryLabelmapFromArray(
  #     final_mask.astype("uint8"),
  #     segmentationNode, segmentId, volumeNode
  # )
  # segmentationNode.CreateClosedSurfaceRepresentation()
  # print("Segmentation successfully refined with bounding boxes\n")

    # ============================================================
    # ======= PART 3.5 (Model Registration) - 09/23/2025 =========
    # ============================================================
    #print("here ######################")
    #checkCenter(segmentationNode,volumeNode)
    #print("out ######################")


    print("Converting segmentation to model for registration...")
    segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
    segmentation_model_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', "TubeSegmentationModel")
    slicer.modules.segmentations.logic().ExportSegmentToRepresentationNode(segment, segmentation_model_node)
    print(f"Successfully created model: {segmentation_model_node.GetName()}")

    # Load the source model (the ideal tube shape)
    # moving_model_node = slicer.util.loadModel('tube-Body001.stl')
    moving_model_node = slicer.util.loadModel(stl_path) 

    output_transform_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "RegistrationTransform")

    # Run Registration
    reg_logic = ModelRegistrationLogic()
    success = reg_logic.run(
        inputSourceModel=moving_model_node,
        inputTargetModel=segmentation_model_node,
        outputSourceToTargetTransform=output_transform_node,
        transformType=0,  
        numIterations=200
    )

    if success:
        print("Registration completed successfully!")
        #moving_model_node.SetAndObserveTransformNodeID(output_transform_node.GetID())
        #mean_distance = reg_logic.ComputeMeanDistance(moving_model_node, segmentation_model_node, output_transform_node)
        #print(f"Mean distance after registration: {mean_distance:.4f}")
    else:
        print("Registration failed.")
        return None, float('inf')


    # ============================================================
    # =============== PART 4 (Centerline and Projection) ===============
    # ============================================================

    print("Extracting centerline from registered model...")
    # use segmentation or registered model
    point_coordinates = slicer.util.arrayFromModelPoints(moving_model_node) # registered model 
    #point_coordinates = slicer.util.arrayFromModelPoints(segmentation_model_node) #segmentation


    pts = slicer.util.arrayFromModelPoints(moving_model_node).astype(float)

    # PCA
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    eigvals, eigvecs = np.linalg.eigh(cov)      # ascending order
    axis = eigvecs[:, np.argmax(eigvals)]       # longest-axis direction (unit vector)

    # Extent along that axis -> endpoints
    proj = (pts - center) @ axis
    p1 = center + proj.min() * axis
    p2 = center + proj.max() * axis
    print("Length (mm):", np.linalg.norm(p2 - p1))

    # Create the line
    line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "LongAxis")
    line.AddControlPoint(p1.tolist())
    line.AddControlPoint(p2.tolist())

    # Option A: extend 100 mm beyond p2 (one end)
    q1, q2 = p1, p1 - 100 * axis


    extLine = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "LongAxisExtended")
    extLine.AddControlPoint(q1.tolist())
    extLine.AddControlPoint(q2.tolist())
    print("New length (mm):", np.linalg.norm(q2 - q1))


    pointsNode = extLine

    # Output model
    modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "NeedlePath")
    modelNode.CreateDefaultDisplayNodes()

    # Markups-to-model parameter node
    m2m = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsToModelNode", "MarkupsToModel")
    m2m.SetAndObserveInputNodeID(pointsNode.GetID())
    m2m.SetAndObserveOutputModelNodeID(modelNode.GetID())

    m2m.SetModelType(m2m.ClosedSurface)
    m2m.SetCleanMarkups(True)
    m2m.SetButterflySubdivision(True)
    m2m.SetDelaunayAlpha(0.0)      # 0 = no alpha limit (convex-ish result)
    m2m.SetConvexHull(False)

    m2m.SetAutoUpdateOutput(True)  # regenerate automatically when points move

    # Force an update now
    slicer.modules.markupstomodel.logic().UpdateOutputModel(m2m)

    #registered
    modelNode.SetAndObserveTransformNodeID(output_transform_node.GetID())
    moving_model_node.SetAndObserveTransformNodeID(output_transform_node.GetID())
    pointsNode.SetAndObserveTransformNodeID(output_transform_node.GetID())
    displayNode = modelNode.GetDisplayNode()
    displayNode.SetVisibility(True)       # overall visibility
    displayNode.SetVisibility2D(True)     # show intersection in slice views

    bottom_points = point_coordinates[point_coordinates[:, 2] == 0]
    top_points    = point_coordinates[point_coordinates[:, 2] == 65]

    print("##########")
    print(bottom_points)
    print(top_points)

    bottom_center_local = bottom_points.mean(axis=0)
    top_center_local    = top_points.mean(axis=0)

    # Create a line node with the local coordinates
    lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "CenterLine")
    lineNode.AddControlPointWorld(vtk.vtkVector3d(*bottom_center_local))
    lineNode.AddControlPointWorld(vtk.vtkVector3d(*top_center_local))

    # Apply the registration transform to the line to move it into world space
    lineNode.SetAndObserveTransformNodeID(output_transform_node.GetID())

    lineNode.HardenTransform()
    
    disp = lineNode.GetDisplayNode()
    disp.SetColor(1, 0, 1)  
    disp.SetLineWidth(2.0)
    disp.SetGlyphScale(2.5)

    print(f"Successfully created centerline from registered model.")

    # Clean up the scene for better visualization
    segmentation_model_node.GetDisplayNode().SetVisibility(0)
    segmentationNode.GetDisplayNode().SetVisibility(0)
    moving_model_node.GetDisplayNode().SetVisibility2D(True)

    # Line endpoints from the markups used to build the model (world coords, so transforms count)
    a, b = [0.0]*3, [0.0]*3
    pointsNode.GetNthControlPointPositionWorld(0, a)
    pointsNode.GetNthControlPointPositionWorld(pointsNode.GetNumberOfControlPoints() - 1, b)

    # Query point
    p = [0.0]*3
    target.GetNthControlPointPositionWorld(0, p)

    d, closest, t = point_to_line(p, a, b)
    print(f"Distance to line: {d:.2f} mm (closest point {t:.1f} mm from the first point)")


    results = {
        "volumeNode": volumeNode,
        "segmentationNode": segmentationNode,
        "segmentationModelNode": segmentation_model_node,
        "registeredModelNode": moving_model_node,
        "transformNode": output_transform_node,
        "centerlineNode": lineNode,
        "final_mask": d
    }

    return results,0.0

