import glob as gb
# folder = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/exams/*"
folder = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/exams/王佳荣-2018期货从业-期货基础知识-精（改WORD）/*"

# folder = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/exams/王佳荣-2018期货从业-期货基础知识-精（改WORD)/*.doc"
path = gb.glob(folder)
# print ("path:",  path)
for path in path:
    print (path)
