import SimpleITK as sitk

def segment_image(image):
    thresh_filter = sitk.OtsuThresholdImageFilter()
    thresh_filter.SetInsideValue(1)
    thresh_filter.SetOutsideValue(0)
    segmented_img = thresh_filter.Execute(image)
    thresh_value = thresh_filter.GetThreshold()
    return segmented_img, thresh_value


def affine_align_image(fixed_img_path, moving_img_path, verbose=True):
    fixed_img = sitk.ReadImage(fixed_img_path, sitk.sitkFloat32)
    moving_img = sitk.ReadImage(moving_img_path, sitk.sitkFloat32)

    fixed_img_seg, fixed_img_seg_thresh = segment_image(fixed_img)
    moving_img_seg, moving_img_seg_thresh = segment_image(moving_img)
    
    initial_transform = sitk.CenteredTransformInitializer(fixed_img, 
                                                          moving_img, 
                                                          sitk.Euler2DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    # initial_transform = sitk.CenteredTransformInitializer(fixed_img, 
    #                                                       moving_img, 
    #                                                       sitk.AffineTransform(fixed_img.GetDimension()), 
    #                                                       sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    registration_method = sitk.ImageRegistrationMethod()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Similarity metric settings.
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=10)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.8)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    registration_method.SetMetricFixedMask(fixed_img_seg)
    registration_method.SetMetricMovingMask(moving_img_seg)
    # Optimizer settings.
    # registration_method.SetOptimizerAsLBFGS2(numberOfIterations=1000)
    registration_method.SetOptimizerAsExhaustive(numberOfSteps=[10,10,0])

    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [8,4,2,1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0,0])

    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [16, 8, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])


    final_transform = registration_method.Execute(fixed_img, moving_img)

    # Always check the reason optimization terminated.
    if verbose:
        print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
        print('Transform parameters: ', final_transform.GetParameters())

    moving_resampled = sitk.Resample(moving_img, fixed_img, final_transform, sitk.sitkNearestNeighbor, 1.0, moving_img.GetPixelID())
    return moving_resampled, final_transform, fixed_img_seg, moving_img_seg