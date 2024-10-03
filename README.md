


https://github.com/user-attachments/assets/8780ec90-f2dd-4f42-8406-f52af117cfdb


LAFAN Re-solved
===============

This is a re-solve of the [Ubisoft La Forge Animation Dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) from the original [c3d marker data](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/tree/master/c3d) that was released alongside it.

I have re-solved this marker data using MotionBuilder onto a new actor and new skeleton to the best of my (limited) MotionBuilder abilities to try and resolve some of the data quality issues that were there in the original bvh release.

More specifically this version:

* Has been re-solved at 60fps instead of 30fps.
* Has been retargeted onto a common skeleton (used by the [ZeroEGGS](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS) dataset) to make it compatible with other datasets.
* Contains toe motion that was missing from the original bvh release.
* Contains some degree of finger motion (solved from a single finger marker) that was missing from the original bvh release.
* Includes fbx files and a basic skinned character, not just bvh files.
* Has more care taken during the retargeting to preserve overall motion and posing.

I've also made available the MotionBuilder Actor and Character set-up used for the solve, which can be found under the `subjects` folder - as well as the two quick and dirty MotionBuilder scripts I used to automate the data export process `export_fbx.py` and `export_bvh.py`. The separate bvh export script is required due to MotionBuilder bvh export [accuracy issues](https://twitter.com/anorangeduck/status/1805351572672491569).

The finger motion is still far from perfect and sometimes is quite jittery but at least there is something there and it isn't terrible given it is solved from only a single marker per hand. If anyone is able to tweak the actors and character in the `subjects` folder to get a better solve I would welcome it.

The skinned character mesh is available in `Geno.fbx` and is free for non-commercial research use.

This dataset is compatible with:

* [zeroeggs-retarget](https://github.com/orangeduck/zeroeggs-retarget)
* [motorica-retarget](https://github.com/orangeduck/motorica-retarget)

Download
========

* [BVH Data](https://theorangeduck.com/media/uploads/Geno/lafan1-resolved/bvh.zip)
* [FBX Data](https://theorangeduck.com/media/uploads/Geno/lafan1-resolved/fbx.zip)

License
=======

This version of the data is licensed under the [same terms](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt) as the original dataset.

This means this data is NOT licensed for commercial use.


Citations
=========

When mentioning this database in an academic paper or other publication please cite the following publication as requested by the original repository:

```
@article{harvey2020robust,
author    = {Félix G. Harvey and Mike Yurick and Derek Nowrouzezahrai and Christopher Pal},
title     = {Robust Motion In-Betweening},
booktitle = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH)},
publisher = {ACM},
volume    = {39},
number    = {4},
year      = {2020}
}
```
