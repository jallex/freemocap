import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import cv2 
from scipy.signal import savgol_filter

import copy
from pathlib import Path

#RICH CONSOLE STUFF
from rich import pretty
pretty.install() #makes all print statement output pretty
from rich import inspect
from rich.console import Console
console = Console()  
from rich.traceback import install as rich_traceback_install
from rich.markdown import Markdown

#colors from Taylor Davis branding - 
humon_dark = np.array([37, 67, 66])/255
humon_green = np.array([53, 93, 95])/255
humon_blue = np.array([14, 90, 253])/255
humon_red = np.array([217, 61, 67])/255


def PlaySkeletonAnimation(
    session=None,
    vidType=1,
    startFrame=0,
    azimuth=-90,
    elevation=-61,
    numCams=4,
    useCams = None,
    useOpenPose=True,
    useMediaPipe=False,
    useDLC=False,
    recordVid = True
    ):

#create figure
    fig = plt.figure(dpi=200)
    plt.ion()

#create axes
    ax3d = fig.add_subplot(projection='3d')
    ax3d.set_position([-.065, .35, .7, .7]) # [left, bottom, width, height])
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])

    ax3d.tick_params(length=0) # WHY DOESNT THIS WORK? I HATE THOSE TICK MARKS >:(
    

    #Loading OpenPoseData
    try:
        skel_fr_mar_dim = np.load(session.dataArrayPath / 'openPoseSkel_3d.npy')
        openPoseData_nCams_nFrames_nImgPts_XYC = np.load(session.dataArrayPath / 'openPoseData_2d.npy')
    except:
        console.warning('No openPose data found.  This iteration requires OpenPose data')

    smoothThese = np.arange(0, skel_fr_mar_dim.shape[1])

    for mm in smoothThese:
        if mm > 24 and mm < 67: #don't smooth the hands, or they disappear! :O
            pass
        else:
            for dim in range(skel_fr_mar_dim.shape[2]):
                skel_fr_mar_dim[:,mm,dim] = savgol_filter(np.nan_to_num(skel_fr_mar_dim[:,mm,dim], 5, 3))

    figure_data = dict()

    skel_trajectories = [skel_fr_mar_dim[:,markerNum,:] for markerNum in range(skel_fr_mar_dim.shape[1])]
    figure_data['skel_trajectories|mar|fr_dim'] = skel_trajectories
    figure_data['skel_fr_mar_dim'] = skel_fr_mar_dim
    dict_of_openPoseSegmentIdx_dicts, dict_of_skel_lineColor = formatOpenPoseStickIndices() #these will help us draw body and hands stick figures

    # %% Buildings Artists from the Segments?
    def build_3d_segment_artist_dict(data_fr_mar_dim,
                                     dict_of_list_of_segment_idxs, 
                                     segColor = 'k', 
                                     lineWidth = 1, 
                                     lineStyle = '-', 
                                     markerType = None, 
                                     marSize = 12, 
                                     markerEdgeColor = 'k',):       
        """ 
        Builds a dictionary of line artists for each 3D body segment.
        """       
        segNames = list(dict_of_list_of_segment_idxs)



        dict_of_artist_objects = dict()
        for segNum, segName in enumerate(segNames):

            #determine color of segment, based on class of 'segColor' input
            if isinstance(segColor, str):
                thisRGBA = segColor
            elif isinstance(segColor, np.ndarray):
                thisRGBA = segColor
            elif isinstance(segColor, dict):
                thisRGBA = segColor[segName]
            elif isinstance(segColor, list):
                try:
                    thisRGBA = segColor[segNum]
                except:
                    print('Not enough colors provided, using Black instead')
                    thisRGBA = 'k'
            else:
                thisRGBA = 'k'

            if isinstance(segName, str):
                idxsOG = dict_of_list_of_segment_idxs[segName]
            else:
                idxsOG

            if isinstance(idxsOG, int) or isinstance(idxsOG, float): 
                idxs = [idxsOG]
            elif isinstance(idxsOG, dict):
                idxs = idxsOG[0]
            else:
                idxs = idxsOG.copy()


            dict_of_artist_objects[segName]  = ax3d.plot(
                                                    data_fr_mar_dim[startFrame, idxs ,0], 
                                                    data_fr_mar_dim[startFrame, idxs ,1], 
                                                    data_fr_mar_dim[startFrame, idxs ,2],
                                                    linestyle=lineStyle,
                                                    linewidth=lineWidth,
                                                    markerSize = marSize,
                                                    marker = markerType,
                                                    color = thisRGBA,
                                                    markeredgecolor = markerEdgeColor,
                                                    )[0]
        return dict_of_artist_objects


    matplotlib_artist_objs = dict()
    matplotlib_artist_objs['body'] = build_3d_segment_artist_dict(skel_fr_mar_dim, dict_of_openPoseSegmentIdx_dicts['body'], segColor = dict_of_skel_lineColor)
    matplotlib_artist_objs['rHand'] = build_3d_segment_artist_dict(skel_fr_mar_dim, dict_of_openPoseSegmentIdx_dicts['rHand'], segColor=np.append(humon_red, 1), markerType='.', markerEdgeColor = humon_red, lineWidth=1, marSize = 2)
    matplotlib_artist_objs['lHand'] = build_3d_segment_artist_dict(skel_fr_mar_dim, dict_of_openPoseSegmentIdx_dicts['lHand'], segColor=np.append(humon_blue, 1), markerType='.', markerEdgeColor = humon_blue, lineWidth=1, marSize = 2)
    matplotlib_artist_objs['face'] = build_3d_segment_artist_dict(skel_fr_mar_dim, dict_of_openPoseSegmentIdx_dicts['face'], segColor='k', lineWidth=.5)
 

    skel_dottos = []
    for mm in range(67): #getcher dottos off my face!
        thisTraj = skel_fr_mar_dim[:, mm, :]
        if mm==15:
            col = 'r'
            markerSize = 2
        elif mm == 16:
            col = 'b'
            markerSize = 2
        else:
            col = 'k'
            markerSize = 1

        thisDotto =ax3d.plot(thisTraj[0, 0:1], thisTraj[1, 0:1], thisTraj[2, 0:1][0],'.', markersize=markerSize, color = col)
        skel_dottos.append(thisDotto[0])

    matplotlib_artist_objs['skel_dottos'] = skel_dottos

    numFrames = skel_fr_mar_dim.shape[0]
   

    mx = np.nanmean(skel_fr_mar_dim[int(numFrames/2),:,0])
    my = np.nanmean(skel_fr_mar_dim[int(numFrames/2),:,1])
    mz = np.nanmean(skel_fr_mar_dim[int(numFrames/2),:,2])


    axRange = 500#session.board.square_length * 10

    # Setting the axes properties
    ax3d.set_xlim3d([mx-axRange, mx+axRange])
    ax3d.set_ylim3d([my-axRange, my+axRange])
    ax3d.set_zlim3d([mz-axRange-1600, mz+axRange-1600])
    
    fig.suptitle("-The FreeMoCap Project-", fontsize=14)

    ax3d.view_init(azim=azimuth, elev=elevation)

# %%  Make Video Axes
    syncedVidPathListAll = list(sorted(session.syncedVidPath.glob('*.mp4')))
    
    #remove a few vids, 6 is too many! NOTE - this is kinda hardcoded for the 20-07-2021 release video
    if session.sessionID == 'sesh_21-07-08_131030':
        useCams = [0,1,2,3]
    #     delTheseVids = [4,1]
    
    if useCams: #JSM NOTE _ This might not work at all lol 
        syncedVidPathList = [syncedVidPathListAll[camNum] for camNum in useCams]
        dlcData_nCams_nFrames_nImgPts_XYC = dlcData_nCams_nFrames_nImgPts_XYC[useCams, :, :, :]
        openPoseData_nCams_nFrames_nImgPts_XYC = openPoseData_nCams_nFrames_nImgPts_XYC[useCams, :, :, :]
    else:
        syncedVidPathList  = syncedVidPathListAll.copy()

    vidAxesList = []
    vidAristList = []
    vidCapObjList = []

    list_of_vidOpenPoseArtistdicts = []
    vidDLCArtistList = []
    
    vidAx_positions = []

    left = .45
    bottom = 0.05
    vidWidth = .38
    vidHeight = vidWidth
    widthScale = .6
    heightScale = 1.2

    vidAx_positions.append([
        left, 
        bottom, 
        vidWidth, 
        vidHeight])

    vidAx_positions.append([
        left+vidWidth*widthScale, 
        bottom, 
        vidWidth, 
        vidHeight])

    vidAx_positions.append([
        left, 
        bottom+vidHeight*heightScale, 
        vidWidth, 
        vidHeight])

    vidAx_positions.append([
        left+vidWidth*widthScale,
        bottom+vidHeight*heightScale, 
        vidWidth, 
        vidHeight])

    def build_2d_segment_artist_dict(vidNum, data_nCams_nFrames_nImgPts_XYC, dict_of_list_of_segment_idxs, segColor = 'k', lineWidth = 1, lineStyle = '-'):       
        """ 
        Builds a dictionary of line artists for each body 2d segment.
        """       
        segNames = list(dict_of_list_of_segment_idxs)

        dict_of_artist_objects = dict()
        for segNum, segName in enumerate(segNames):

            #determine color of segment, based on class of 'segColor' input
            if isinstance(segColor, str):
                thisRGBA = segColor
            elif isinstance(segColor, np.ndarray):
                thisRGBA = segColor
            elif isinstance(segColor, dict):
                thisRGBA = segColor[segName].copy()
                thisRGBA[-1] = .75
            elif isinstance(segColor, list):
                try:
                    thisRGBA = segColor[segNum]
                except:
                    print('Not enough colors provided, using Black instead')
                    thisRGBA = 'k'
            else:
                thisRGBA = 'k'

            xData = data_nCams_nFrames_nImgPts_XYC[vidNum, startFrame, dict_of_list_of_segment_idxs[segName],0]
            yData = data_nCams_nFrames_nImgPts_XYC[vidNum, startFrame, dict_of_list_of_segment_idxs[segName],1]

            # #make NaN's invisible (i thought they already would be but???!!!!)
            # thisRGBAall = np.tile(thisRGBA,(len(xData),1))
            # thisRGBAall[np.isnan(xData),3] = 0
            
            xDataMasked  = np.ma.masked_where(xData, np.isnan(xData))
            yDataMasked  = np.ma.masked_where(yData, np.isnan(yData))

            dict_of_artist_objects[segName]  = thisVidAxis.plot(
                                                    xDataMasked,
                                                    yDataMasked,
                                                    linestyle=lineStyle,
                                                    linewidth=lineWidth,
                                                    color = thisRGBA,
                                                    )[0]
        return dict_of_artist_objects

    for vidSubplotNum, thisVidPath in enumerate(syncedVidPathList):
            #make subplot for figure (and set position)
            thisVidAxis = fig.add_subplot(
                                        position=vidAx_positions[vidSubplotNum], 
                                        label="Vid_{}".format(str(vidSubplotNum)),
                                        ) 

            thisVidAxis.set_axis_off()

            vidAxesList.append(thisVidAxis)

            #create video capture object
            thisVidCap = cv2.VideoCapture(str(thisVidPath))
            

            #create artist object for each video 
            success, image  = thisVidCap.read()

            assert success==True, "{} - failed to load an image".format(thisVidPath.stem) #make sure we have a frame

            vidAristList.append(thisVidAxis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            vidCapObjList.append(thisVidCap)



            if useOpenPose:
                vidOpenPoseArtist_dict = dict()            
                vidOpenPoseArtist_dict['body'] = build_2d_segment_artist_dict(vidSubplotNum, openPoseData_nCams_nFrames_nImgPts_XYC, dict_of_openPoseSegmentIdx_dicts['body'], segColor = dict_of_skel_lineColor)
                vidOpenPoseArtist_dict['rHand'] = build_2d_segment_artist_dict(vidSubplotNum, openPoseData_nCams_nFrames_nImgPts_XYC, dict_of_openPoseSegmentIdx_dicts['rHand'], segColor=np.append(humon_red, .75), lineWidth=.5)
                vidOpenPoseArtist_dict['lHand'] = build_2d_segment_artist_dict(vidSubplotNum, openPoseData_nCams_nFrames_nImgPts_XYC, dict_of_openPoseSegmentIdx_dicts['lHand'], segColor=np.append(humon_blue, .75), lineWidth=.5)
                vidOpenPoseArtist_dict['face'] = build_2d_segment_artist_dict(vidSubplotNum, openPoseData_nCams_nFrames_nImgPts_XYC, dict_of_openPoseSegmentIdx_dicts['face'], segColor = np.array([1.,1.,1.,1]), lineWidth=.25)
                list_of_vidOpenPoseArtistdicts.append(vidOpenPoseArtist_dict)

            if session.useDLC:
                vidDLCArtistList.append(thisVidAxis.plot(
                                        dlcData_nCams_nFrames_nImgPts_XYC[vidSubplotNum,startFrame,:,0], 
                                        dlcData_nCams_nFrames_nImgPts_XYC[vidSubplotNum,startFrame,:,1],
                                        linestyle='none',
                                        marker = '.',
                                        markersize = 1,
                                        color='w',
                                        markerfacecolor='none' ))
    

def formatOpenPoseStickIndices():
    """
    generate dictionary of arrays, each containing the 'connect-the-dots' order to draw a given body segment
    
    returns:
    openPoseBodySegmentIds= a dictionary of arrays containing indices of individual body segments (Note, a lot of markerless mocap comp sci types like to say 'pose' instead of 'body'. They also use 'pose' to refer to camera 6 DoF position sometimes. Comp sci is frustrating like that lol)
    openPoseHandIds = a dictionary of arrays containing indices of individual hand segments, along with offset to know where to start in the 'skel_fr_mar_dim.shape[1]' part of the array
    dict_of_skel_lineColor = a dictionary of arrays, each containing the color (RGBA) to use for a given body segment
    """
    dict_of_openPoseSegmentIdx_dicts = dict()

    #make body dictionary
    openPoseBodySegmentIds = dict()
    openPoseBodySegmentIds['head'] = [17, 15, 0, 1,0, 16, 18, ]
    openPoseBodySegmentIds['spine'] = [1,8,5,1, 2, 12, 8, 9, 5, 1, 2, 8]
    openPoseBodySegmentIds['rArm'] = [1, 2, 3, 4, ]
    openPoseBodySegmentIds['lArm'] = [1, 5, 6, 7, ]
    openPoseBodySegmentIds['rLeg'] = [8, 9, 10, 11, 22, 23, 11, 24, ]
    openPoseBodySegmentIds['lLeg'] = [8,12, 13, 14, 19, 20, 14, 21,]
    dict_of_openPoseSegmentIdx_dicts['body'] = openPoseBodySegmentIds


    dict_of_skel_lineColor = dict()
    dict_of_skel_lineColor['head'] = np.append(humon_dark, 0.5)
    dict_of_skel_lineColor['spine'] = np.append(humon_dark, 1)
    dict_of_skel_lineColor['rArm'] = np.append(humon_red, 1)
    dict_of_skel_lineColor['lArm'] = np.append(humon_blue, 1)
    dict_of_skel_lineColor['rLeg'] = np.append(humon_red, 1)
    dict_of_skel_lineColor['lLeg'] = np.append(humon_blue, 1)


    # Make some handy maps ;D
    openPoseHandIds = dict()
    rHandIDstart = 25
    lHandIDstart = rHandIDstart + 21

    openPoseHandIds['thumb'] = np.array([0, 1, 2, 3, 4,  ]) 
    openPoseHandIds['index'] = np.array([0, 5, 6, 7, 8, ])
    openPoseHandIds['bird']= np.array([0, 9, 10, 11, 12, ])
    openPoseHandIds['ring']= np.array([0, 13, 14, 15, 16, ])
    openPoseHandIds['pinky'] = np.array([0, 17, 18, 19, 20, ])
    

    rHand_dict = copy.deepcopy(openPoseHandIds.copy()) #copy.deepcopy() is necessary to make sure the dicts are independent of each other
    lHand_dict = copy.deepcopy(rHand_dict)

    for key in rHand_dict: 
        rHand_dict[key] += rHandIDstart 
        lHand_dict[key] += lHandIDstart 

    dict_of_openPoseSegmentIdx_dicts['rHand'] = rHand_dict
    dict_of_openPoseSegmentIdx_dicts['lHand'] = lHand_dict

    
    #how to face --> :D <--
    openPoseFaceIDs = dict()
    faceIDStart = 67
    #define face parts
    openPoseFaceIDs['jaw'] = np.arange(0,16) + faceIDStart 
    openPoseFaceIDs['rBrow'] = np.arange(17,21) + faceIDStart
    openPoseFaceIDs['lBrow'] = np.arange(22,26) + faceIDStart
    openPoseFaceIDs['noseRidge'] = np.arange(27,30) + faceIDStart
    openPoseFaceIDs['noseBot'] = np.arange(31,35) + faceIDStart
    openPoseFaceIDs['rEye'] = np.concatenate((np.arange(36,41), [36])) + faceIDStart
    openPoseFaceIDs['lEye'] = np.concatenate((np.arange(42,47), [42])) + faceIDStart    
    openPoseFaceIDs['upperLip'] = np.concatenate((np.arange(48,54), np.flip(np.arange(60, 64)), [48])) + faceIDStart
    openPoseFaceIDs['lowerLip'] = np.concatenate(([60], np.arange(64,67), np.arange(54, 59), [48], [60])) + faceIDStart
    openPoseFaceIDs['rPupil'] = np.array([68]) + faceIDStart
    openPoseFaceIDs['lPupil'] = np.array([69]) + faceIDStart #nice

    dict_of_openPoseSegmentIdx_dicts['face'] = openPoseFaceIDs
    
    return dict_of_openPoseSegmentIdx_dicts, dict_of_skel_lineColor


