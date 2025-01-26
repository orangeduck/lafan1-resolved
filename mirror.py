import re, os
import numpy as np
import click

def bvh_load(filename, order=None):

    channelmap = {
        'Xrotation': 'x',
        'Yrotation': 'y',
        'Zrotation': 'z'
    }

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    return {
        'rotations': rotations,
        'positions': positions,
        'offsets': offsets,
        'parents': parents,
        'names': names,
        'order': order,
        'frametime': frametime,
    }
    
    
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
    
def fast_cross(a, b):
    return np.concatenate([
        a[...,1:2]*b[...,2:3] - a[...,2:3]*b[...,1:2],
        a[...,2:3]*b[...,0:1] - a[...,0:1]*b[...,2:3],
        a[...,0:1]*b[...,1:2] - a[...,1:2]*b[...,0:1]], axis=-1)

def quat_inv(q):
    return np.asarray([1, -1, -1, -1], dtype=np.float32) * q

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

def quat_mul_inv(x, y):
    return quat_mul(x, quat_inv(y))

def quat_mul_vec(q, x):
    t = 2.0 * fast_cross(q[..., 1:], x)
    return x + q[..., 0][..., np.newaxis] * t + fast_cross(q[..., 1:], t)

def quat_inv_mul_vec(q, x):
    return quat_mul_vec(inv(q), x)
    
def quat_unroll(x):
    y = x.copy()
    for i in range(1, len(x)):
        d0 = np.sum( y[i] * y[i-1], axis=-1)
        d1 = np.sum(-y[i] * y[i-1], axis=-1)
        y[i][d0 < d1] = -y[i][d0 < d1]
    return y

def quat_fk(lrot, lpos, parents):
    
    gp, gr = [lpos[...,:1,:]], [lrot[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(quat_mul    (gr[parents[i]], lrot[...,i:i+1,:]))
        
    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)

def quat_ik(grot, gpos, parents):
    
    return (
        np.concatenate([
            grot[...,:1,:],
            quat_mul(quat_inv(grot[...,parents[1:],:]), grot[...,1:,:]),
        ], axis=-2),
        np.concatenate([
            gpos[...,:1,:],
            quat_mul_vec(
                quat_inv(grot[...,parents[1:],:]),
                gpos[...,1:,:] - gpos[...,parents[1:],:]),
        ], axis=-2))
    
def quat_from_angle_axis(angle, axis):
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q
    
def quat_from_euler(e, order='zyx'):
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = quat_from_angle_axis(e[..., 0], axis[order[0]])
    q1 = quat_from_angle_axis(e[..., 1], axis[order[1]])
    q2 = quat_from_angle_axis(e[..., 2], axis[order[2]])

    return quat_mul(q0, quat_mul(q1, q2))    

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

    
@click.command()
@click.argument('input')
@click.argument('output')
@click.option('-e', '--enforce-offsets', 'enforce_offsets', type=bool, default=True, help='Enforce joint offsets are kept the same. Introduces slight error in the positioning of some joints (~1cm) but preserves local translations exactly.')
def mirror(input, output, enforce_offsets):
    
    if not os.path.exists(input):
        raise Exception('Input file "%s" does not exist.' % input)
    
    if not input.lower().endswith('.bvh'):
        raise Exception('Input file not a bvh file.')
    
    print('Loading "%s"...' % input)
    
    # Load
    
    bvh_data = bvh_load(input)
    Xloc_pos = bvh_data['positions'].astype(np.float32)
    Xloc_rot = quat_unroll(quat_from_euler(np.radians(bvh_data['rotations']), order=bvh_data['order'])).astype(np.float32)
    
    # Find Mirror Bones
    
    mirror_bones = []
    for ni, n in enumerate(bvh_data['names']):
        if 'Right' in n and n.replace('Right', 'Left') in bvh_data['names']:
            mirror_bones.append(bvh_data['names'].index(n.replace('Right', 'Left')))
        elif 'Left' in n and n.replace('Left', 'Right') in bvh_data['names']:
            mirror_bones.append(bvh_data['names'].index(n.replace('Left', 'Right')))
        else:
            mirror_bones.append(ni)
    
    mirror_bones = np.array(mirror_bones)
    
    # Mirror
    
    Xglo_rot, Xglo_pos = quat_fk(Xloc_rot, Xloc_pos, bvh_data['parents'])
    Xglo_pos = np.array([-1, 1, 1]) * Xglo_pos[:,mirror_bones]
    Xglo_rot = np.array([1, 1, -1, -1]) * Xglo_rot[:,mirror_bones]
    Xloc_rot, Xloc_pos = quat_ik(Xglo_rot, Xglo_pos, bvh_data['parents'])
    
    # Save
    
    bvh_data['positions'] = Xloc_pos
    bvh_data['rotations'] = np.degrees(quat_to_euler(Xloc_rot))
    
    if enforce_offsets:
        bvh_data['positions'][:,1:] = bvh_data['offsets'][1:]
    else:
        bvh_data['offsets'][1:] = Xloc_pos[0,1:]
    
    print('Saving "%s"...' % output)

    bvh_save(output, bvh_data)
    
    
if __name__ == '__main__':
    mirror()
    
    