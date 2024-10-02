import os

def select_branch(node):
    if isinstance(node, pyfbsdk.FBModel):
        
        node.Selected = True
        
        for child in node.Children:
            select_branch(child)

def deselect_all():
    selected_models = FBModelList()
    FBGetSelectedModels(selected_models, None, True)
    for select in selected_models:
        select.Selected = False;

subjects = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5']
bvh_files = os.listdir('C:/Data/lafan1_bvh')

for subject in subjects:
    
    categories = set([f.split('_')[0] for f in bvh_files if subject in f])
    
    for category in categories:
        
        subject_file = 'C:/Dev/lafan1-resolved/subjects/'+subject+'.fbx'    
        c3d_file = 'C:/Data/lafan1_c3d/'+category+'.c3d'
        output_fbx_file = 'C:/Dev/lafan1-resolved/fbx/'+category+'_'+subject+'.fbx'
        
        # Modifications from https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/c3d/c3d_readme.txt
        if subject == 'subject1' and category == 'fallAndGetUp3':
            c3d_file = 'C:/Data/lafan1_c3d/pushAndFall2.c3d'
        
        if subject == 'subject2' and category == 'push1':
            c3d_file = 'C:/Data/lafan1_c3d/pushAndFall2.c3d'
        
        if subject == 'subject1' and category == 'walk4':
            c3d_file = 'C:/Data/lafan1_c3d/fallAndGetUp2.c3d'
        
        print('Creating %s' % output_fbx_file)
        
        options = FBFbxOptions(True)
        options.TakeSpan = FBTakeSpanOnLoad.kFBLeaveAsIs
        
        FBApplication().FileOpen(subject_file)
        FBApplication().FileMerge('C:/Dev/lafan1-resolved/Geno.fbx', False, options)
        
        FBSystem().Scene.Characters[0].InputActor = FBSystem().Scene.Actors[0]
        FBSystem().Scene.Characters[0].InputType = FBCharacterInputType.kFBCharacterInputActor
        FBSystem().Scene.Characters[0].ActiveInput = True
        
        FBSystem().Scene.Characters[0].PropertyList.Find('ReachActorLeftWrist').Data = 100.0
        FBSystem().Scene.Characters[0].PropertyList.Find('ReachActorLeftWristRotation').Data = 100.0
        FBSystem().Scene.Characters[0].PropertyList.Find('ReachActorRightWrist').Data = 100.0
        FBSystem().Scene.Characters[0].PropertyList.Find('ReachActorRightWristRotation').Data = 100.0
        FBSystem().Scene.Characters[0].PropertyList.Find('ReachActorLeftFingerBase').Data = 100.0
        FBSystem().Scene.Characters[0].PropertyList.Find('ReachActorLeftFingerBaseRotation').Data = 100.0
        FBSystem().Scene.Characters[0].PropertyList.Find('ReachActorRightFingerBase').Data = 100.0
        FBSystem().Scene.Characters[0].PropertyList.Find('ReachActorRightFingerBaseRotation').Data = 100.0
        # FBSystem().Scene.Characters[0].PropertyList.Find('ReachActorChest').Data = 100.0
        # FBSystem().Scene.Characters[0].PropertyList.Find('ReachActorChestRotation').Data = 100.0
        
        FBApplication().FileImport(c3d_file, True)
        
        deselect_all()
        select_branch(FBFindModelByLabelName('Hips'))
        
        print('Plotting...')
        
        FBSystem().CurrentTake.PlotTakeOnSelected(FBTime(0,0,0,1))
    
        FBFindModelByLabelName('Geno').Selected = True
        
        save_options = FBFbxOptions(False)
        save_options.SaveSelectedModelsOnly = True
        
        print('Saving FBX...')
        
        FBApplication().FileSave(output_fbx_file, save_options)
        
        # break
    # break