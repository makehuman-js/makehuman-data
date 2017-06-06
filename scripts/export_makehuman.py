import json
import os
import numpy as np
import re
import tempfile
import logging
logger = logging.getLogger('export_makehuman')

class NP_MH_Encoder(json.JSONEncoder):
    """subclass of json encoder to deal with json array  """

    def iterencode(self, o, _one_shot=False):
        # the stops it indenting list items beyond the first level
        list_lvl = 0
        for s in super(NP_MH_Encoder, self).iterencode(o, _one_shot=_one_shot):
            if s.startswith('['):
                list_lvl += 1
                s = s.replace('\n', '').rstrip()
                s = re.sub('\[\s+','[',s)
            elif 0 < list_lvl:
                s = re.sub(r'\n\s+', '', s).rstrip()
                if s and s[-1] == ',':
                    s = s[:-1] + self.item_separator
                elif s and s[-1] == ':':
                    s = s[:-1] + self.key_separator
            if s.endswith(']'):
                list_lvl -= 1
            yield s

    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            # I ideally want to save float32 as ~4dp, not 8
            if obj.dtype == np.float32:
                return np.around(obj.astype(float), decimals=5).tolist()
            else:
                return obj.tolist()
        if isinstance(obj, np.float32):
            return np.around(obj.astype(float), 5).tolist()
        # some makehuman objs
        if str(type(obj)) == "<class 'animation.VertexBoneWeights'>":
            return obj.data
        if str(type(obj)) == "<class 'makehuman.LicenseInfo'>":
            return obj.asDict()
        return super(NP_MH_Encoder, self).default(obj)


def writeMaterial(fp, mat, outFolder):
    # modified from makehuman/shared/wavefront.py
    # ref: https://github.com/mrdoob/three.js/blob/master/examples/js/loaders/MTLLoader.js#L389
    fp.write("\nnewmtl %s\n" % mat.name)

    diff = mat.diffuseColor
    spec = mat.specularColor
    amb = mat.ambientColor
    emi = mat.emissiveColor

    # alpha=0 is necessary for correct transparency in Blender.
    # But may lead to problems with other apps.
    if mat.diffuseTexture:
        alpha = 1
    else:
        alpha = mat.opacity
    fp.write(
        "Kd %.4g %.4g %.4g\n" % (diff.r, diff.g, diff.b) +
        "Ks %.4g %.4g %.4g\n" % (spec.r, spec.g, spec.b) +
        "Ka %.4g %.4g %.4g\n" % (amb.r, amb.g, amb.b) +
        "Ke %.4g %.4g %.4g\n" % (emi.r, emi.g, emi.b) +
        "d %.4g\n" % alpha +
        "wireframe %s\n" % int(mat.wireframe) +
        "Ns %.4g\n" % mat.shininess

    )

    writeTexture(fp, "map_Kd", mat.diffuseTexture, outFolder)
    writeTexture(fp, "map_Ks", mat.specularMapTexture, outFolder)
    writeTexture(fp, "bump", mat.bumpMapTexture, outFolder)
    writeTexture(fp, "map_ao", mat.aoMapTexture, outFolder)
    writeTexture(fp, "map_norm", mat.normalMapTexture, outFolder)
    writeTexture(fp, "disp", mat.displacementMapTexture, outFolder)

    # writeTexture(fp, "mapUV", mat.uvMap, outFolder)
    # writeTexture(fp, "mapTransparency", mat.transparencyMapTexture, outFolder)
    writeTexture(fp, "map_Ns", mat.specularMapTexture, outFolder)
    # map_ka, map_Ns specularMapTexture, map_d,
    #
def material_to_mtl(material, outdir=tempfile.mkdtemp('convert_obj_three'), texdir=None):
    outfile = os.path.join(outdir, material.name + '.mtl')
    if not texdir:
        texdir = outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    outFolder = os.path.realpath(texdir)
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    writeMaterial(open(outfile, 'w'), material, outFolder)
    return outfile


def copyAndCompress(filepath, newpath):
    from PIL import Image
    image = Image.open(filepath)

    if min([image.size[0], image.size[1]]) > 512:
        scaling = 512.0 / min(image.size)
        new_size = (int(image.size[0] * scaling), int(image.size[1] * scaling))
        image = image.resize(new_size, Image.ANTIALIAS)
    else:
        new_size = image.size
    image.save(newpath, optimize=True, quality=95)
    logger.info('save image to %s (resize from %s to %s)' % (newpath, image.size, new_size))


# modified from export.ExportConfig.copyTextureToNewLocation
def copyTextureToNewLocation(filepath, useRelPaths=True, outFolder=None):
    # srcDir = os.path.abspath(os.path.expanduser(os.path.dirname(filepath)))
    filename = os.path.basename(filepath)

    newpath = os.path.abspath(os.path.join(outFolder, 'textures', filename))
    if not os.path.isdir(os.path.dirname(newpath)):
        os.makedirs(os.path.dirname(newpath))

    try:
        copyAndCompress(filepath, newpath)
    except:
        logger.error("Unable to copy \"%s\" -> \"%s\"" % (filepath, newpath))

    if not useRelPaths:
        return newpath
    else:
        relpath = os.path.relpath(newpath, outFolder)
        return str(os.path.normpath(relpath))

# modified from wavefront.writeTexture
def writeTexture(fp, key, filepath, outFolder):
    if not filepath:
        return

    if outFolder:
        newpath = copyTextureToNewLocation(filepath, outFolder=outFolder)
        fp.write("%s %s\n" % (key, newpath))
    else:
        fp.write("%s %s\n" % (key, filepath))


def vertex_weights_to_skin_weights(vertex_weights, skeleton, influencesPerVertex=4):
    """
    Export the skeleton from threejs

    Note that threejs only support 4 skinweights so any connections above this will be discarded.

    Avoid norm will make the 4th weight have the discarded weights but tied to no vertice, this will avoid normalisation

    see makehuman:shared.animation._compileVertexWeights
    """
    # take all the bone, vertex weight mapping and put them in a table
    # vertex_weights= human.getVertexWeights(prxy.getVertexWeights(skeleton.getVertexWeights()))

    if  vertex_weights._nWeights<influencesPerVertex:
        logger.error('influencesPerVertex is greater than the vertex_weights influencesPerVertex')
    compiled_vertex_weights = vertex_weights.compiled(nWeights=influencesPerVertex, skel=skeleton)
    skinIndices = np.array([np.array(list(vw)[:influencesPerVertex]) for vw in compiled_vertex_weights]) # convert to np array
    skinWeights = np.array([np.array(list(vw)[influencesPerVertex:]) for vw in compiled_vertex_weights])
    skinIndices = skinIndices.flatten().tolist()
    skinWeights = np.around(skinWeights.astype(float), 4).flatten().tolist()

    # we have to double the skinIndices since we doubled the amount of bones
    skinIndices = [d*2 for d in skinIndices]

    return skinIndices, skinWeights

def parse_skeleton_bones(skeleton):
    """
    Parse threejs skeleton to form {name:'',pos:[0,0,0],rotq:[0,0,0,1],scl:[1,1,1],parent:-1}

    Makehuman has bone with a head which is offset from the parent and a tail
    that is offset from the head. Threejs bones only have atail so we will
    turn one bone into two. The "bone____head" and "bone".

    Ref: https://github.com/mrdoob/three.js/blob/master/src/objects/SkinnedMesh.js#L34

    """
    bones = []
    for bone in skeleton.getBones():

        # first the head
        bonedef = dict(
            name=bone.name+'____head',
            pos=[0,0,0],
            rotq=[0,0,0,1],
            scl=[1,1,1],
            parent=-1
        )

        if bone.parent:
            # double the index since we turning bone bone into 2
            bonedef["parent"] = bone.parent.index*2+1
        else:
            bonedef["parent"] = -1


        # tail position relative to parent
        if bone.parent:
            bonedef["pos"]=(bone.headPos-bone.parent.tailPos).tolist()
        else:
            bonedef["pos"]=[0,0,0]

        bones.append(bonedef)

        # now the tail
        bonedef = dict(
            name=bone.name,
            pos=bone.tailPos-bone.headPos,
            rotq=[0,0,0,1],
            scl=[1,1,1],
            parent=len(bones)-1 # the head
        )
        bones.append(bonedef)
    return bones
