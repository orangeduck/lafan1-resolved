import os
import sys

sys.path.append('C:\\Users\\daniel.holden\\anaconda3\\envs\\mobu\\lib\\site-packages')
import numpy as np

def bvh_save(filename, data):

    channelmap_inv = {
        'x': 'Xrotation',
        'y': 'Yrotation',
        'z': 'Zrotation',
    }

    rots, poss, offsets, parents = [
        data['rotations'],
        data['positions'],
        data['offsets'],
        data['parents']]
    
    names = data.get('names', ["joint_" + str(i) for i in range(len(parents))])
    order = data.get('order', 'zyx')
    frametime = data.get('frametime', 1.0/60.0)
    
    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %.9g %.9g %.9g\n" % ((t,) + tuple(offsets[0])))
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % 
            (t, channelmap_inv[order[0]], 
                channelmap_inv[order[1]], 
                channelmap_inv[order[2]]))
        jseq = [0]       
        for i in range(len(parents)):
            if parents[i] == 0:
                t, jseq = bvh_save_joint(f, offsets, order, parents, names, t, i, jseq)

        t = t[:-1]
        f.write("%s}\n" % t)
        f.write("MOTION\n")
        f.write("Frames: %i\n" % len(rots))
        f.write("Frame Time: %f\n" % frametime)
        
        for i in range(rots.shape[0]):
            for j in jseq:
                
                f.write("%.9g %.9g %.9g %.9g %.9g %.9g " % (
                    poss[i,j,0], poss[i,j,1], poss[i,j,2], 
                    rots[i,j,0], rots[i,j,1], rots[i,j,2]))

            f.write("\n")
    
def bvh_save_joint(f, offsets, order, parents, names, t, i, jseq):

    jseq.append(i)

    channelmap_inv = {
        'x': 'Xrotation',
        'y': 'Yrotation',
        'z': 'Zrotation',
    }
    
    f.write("%sJOINT %s\n" % (t, names[i]))
    f.write("%s{\n" % t)
    t += '\t'
  
    f.write("%sOFFSET %.9g %.9g %.9g\n" % ((t,) + tuple(offsets[i])))
    
    f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t, 
        channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

    end_site = True
    
    for j in range(len(parents)):
        if parents[j] == i:
            t, jseq = bvh_save_joint(f, offsets, order, parents, names, t, j, jseq)
            end_site = False
    
    if end_site:
        f.write("%sEnd Site\n" % t)
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %.9g %.9g %.9g\n" % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write("%s}\n" % t)
  
    t = t[:-1]
    f.write("%s}\n" % t)
    
    return t, jseq

def quat_abs(x):
    return np.where((np.sum(x * np.array([1, 0, 0, 0], dtype=np.float32), axis=-1) > 0.0)[..., np.newaxis], x, -x)

def quat_inv(x):
    return np.array([1, -1, -1, -1], dtype=np.float32) * x

def quat_mul(x, y):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

def quat_inv_mul(x, y):
    return quat_mul(quat_inv(x), y)

def _fast_cross(a, b):
    o = np.empty(np.broadcast(a, b).shape)
    o[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    o[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    o[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return o

def quat_mul_vec(x, y):
    t = 2.0 * _fast_cross(x[..., 1:], y)
    return y + x[..., 0][..., np.newaxis] * t + _fast_cross(x[..., 1:], t)

def quat_inv_mul_vec(x, y):
    return quat_mul_vec(quat_inv(x), y)

def quat_to_euler(x, order='zyx'):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]

    if order == 'zyx':
        return np.concatenate([
            np.arctan2(2.0 * (x0 * x3 + x1 * x2), 1.0 - 2.0 * (x2 * x2 + x3 * x3)),
            np.arcsin(np.clip(2.0 * (x0 * x2 - x3 * x1), -1.0, 1.0)),
            np.arctan2(2.0 * (x0 * x1 + x2 * x3), 1.0 - 2.0 * (x1 * x1 + x2 * x2)),
        ], axis=-1)
    elif order == 'xzy':
        return np.concatenate([
            np.arctan2(2.0 * (x1 * x0 - x2 * x3), -x1 * x1 + x2 * x2 - x3 * x3 + x0 * x0),
            np.arctan2(2.0 * (x2 * x0 - x1 * x3), x1 * x1 - x2 * x2 - x3 * x3 + x0 * x0),
            np.arcsin(np.clip(2.0 * (x1 * x2 + x3 * x0), -1.0, 1.0))
        ], axis=-1)
    else:
        raise NotImplementedError('Cannot convert to ordering %s' % order)


def quat_from_xform(ts, eps=1e-10):
    qs = np.empty_like(ts[..., :1, 0].repeat(4, axis=-1))

    t = ts[..., 0, 0] + ts[..., 1, 1] + ts[..., 2, 2]

    s = 0.5 / np.sqrt(np.maximum(t + 1, eps))
    qs = np.where((t > 0)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        (0.25 / s)[..., np.newaxis],
        (s * (ts[..., 2, 1] - ts[..., 1, 2]))[..., np.newaxis],
        (s * (ts[..., 0, 2] - ts[..., 2, 0]))[..., np.newaxis],
        (s * (ts[..., 1, 0] - ts[..., 0, 1]))[..., np.newaxis]
    ], axis=-1), qs)

    c0 = (ts[..., 0, 0] > ts[..., 1, 1]) & (ts[..., 0, 0] > ts[..., 2, 2])
    s0 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 0, 0] - ts[..., 1, 1] - ts[..., 2, 2], eps))
    qs = np.where(((t <= 0) & c0)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 2, 1] - ts[..., 1, 2]) / s0)[..., np.newaxis],
        (s0 * 0.25)[..., np.newaxis],
        ((ts[..., 0, 1] + ts[..., 1, 0]) / s0)[..., np.newaxis],
        ((ts[..., 0, 2] + ts[..., 2, 0]) / s0)[..., np.newaxis]
    ], axis=-1), qs)

    c1 = (~c0) & (ts[..., 1, 1] > ts[..., 2, 2])
    s1 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 1, 1] - ts[..., 0, 0] - ts[..., 2, 2], eps))
    qs = np.where(((t <= 0) & c1)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 0, 2] - ts[..., 2, 0]) / s1)[..., np.newaxis],
        ((ts[..., 0, 1] + ts[..., 1, 0]) / s1)[..., np.newaxis],
        (s1 * 0.25)[..., np.newaxis],
        ((ts[..., 1, 2] + ts[..., 2, 1]) / s1)[..., np.newaxis]
    ], axis=-1), qs)

    c2 = (~c0) & (~c1)
    s2 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 2, 2] - ts[..., 0, 0] - ts[..., 1, 1], eps))
    qs = np.where(((t <= 0) & c2)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 1, 0] - ts[..., 0, 1]) / s2)[..., np.newaxis],
        ((ts[..., 0, 2] + ts[..., 2, 0]) / s2)[..., np.newaxis],
        ((ts[..., 1, 2] + ts[..., 2, 1]) / s2)[..., np.newaxis],
        (s2 * 0.25)[..., np.newaxis]
    ], axis=-1), qs)

    return qs


app = FBApplication()
system = FBSystem()
control = FBPlayerControl()

print('Starting...')

filenames = [f[:-4] for f in os.listdir('C:/Dev/lafan1-resolved/fbx') if f.endswith('.fbx')]

for file in filenames:
    
    input_file = 'C:/Dev/lafan1-resolved/fbx/'+file+'.fbx'
    output_file = 'C:/Dev/lafan1-resolved/bvh/'+file+'.bvh'
    
    print('Input %s' % str(input_file))
    print('Output: %s' % str(output_file))
    
    output_path = os.path.split(output_file)[0]
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print('Opening Input %s' % input_file)

    open_options = FBFbxOptions(True)
    app.FileOpen(input_file, False, open_options)    
    
    print('Finding Bones')

    bones = []
    parents = []
    
    def find_bones(node, parent):
        if isinstance(node, FBModel):
            
            par = len(bones)
            bones.append(node)
            parents.append(parent)
            
            for child in node.Children:
                find_bones(child, par)
    
    find_bones(FBFindModelByLabelName('Hips'), -1)
    parents = np.asarray(parents)
    
    start_frame = system.CurrentTake.LocalTimeSpan.GetStart().GetFrame()
    stop_frame = system.CurrentTake.LocalTimeSpan.GetStop().GetFrame() + 1
    
    print('Extracting Bone Data')

    bone_xforms = np.zeros([stop_frame - start_frame, len(bones), 4, 4])
    
    tmp_mat = FBMatrix()

    for i in range(start_frame, stop_frame):
        control.Goto(FBTime(0,0,0,i))
        system.Scene.Evaluate()
        
        for j, bone in enumerate(bones):
            bone.GetMatrix(tmp_mat)
            bone_xforms[i,j] = np.asarray(tmp_mat).reshape([4,4])
    
    control.Goto(FBTime(0,0,0,0))
    
    print('Computing Local Bones')

    bone_global_positions = bone_xforms[:,:,3,:3].copy()
    bone_global_rotations = quat_from_xform(bone_xforms[:,:,:3,:3].transpose([0,1,3,2]))
    
    bone_local_positions = bone_global_positions.copy()
    bone_local_rotations = bone_global_rotations.copy()
    
    bone_local_positions[:,1:] = quat_inv_mul_vec(bone_global_rotations[:,parents[1:]], bone_global_positions[:,1:] - bone_global_positions[:,parents[1:]])
    bone_local_rotations[:,1:] = quat_abs(quat_inv_mul(bone_global_rotations[:,parents[1:]], bone_global_rotations[:,1:]))
    
    print('Saving Bones')
    
    bvh_save(output_file, {
        'positions': bone_local_positions,
        'rotations': np.degrees(quat_to_euler(bone_local_rotations)),
        'offsets': bone_local_positions[0],
        'parents': np.asarray(parents),
        'names': [str(b.Name) for b in bones],
        'framerate': 1.0/60.0,
    })
    
    # break


# app.FileExit(False)