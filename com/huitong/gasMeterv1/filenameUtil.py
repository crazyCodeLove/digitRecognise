#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/12.
"""
import os
import shutil
import re

class FileNameUtil(object):

    @staticmethod
    def fileExisted(filename):
        if filename is None:
            raise ValueError("filename is None")

        return os.path.exists(filename)

    @staticmethod
    def getDirname(basedir, rdirnameList = []):
        """
        根据基目录和子目录名列表，返回该系统拼接的目录
        :param basedir: 基目录
        :param rdirnameList:子目录名以列表的形式给出，不要带系统分隔符，不同操作系统可能不一样。例子：["data","recognise"]
        """
        if type(rdirnameList) is not list:
            raise ValueError("rdirnameList must be a list")
        if len(rdirnameList) == 0:
            raise ValueError("rdirnameList not have value")

        path = basedir
        for each in rdirnameList:
            path = os.path.join(path, each)
        return path

    @staticmethod
    def getFilename(basedirname, filename):
        """
        根据基目录和文件名，返回该系统表示的文件绝对路径名
        :param basedirname: 基目录
        :param filename: 文件名
        """
        return os.path.join(basedirname, filename)

    @staticmethod
    def emptyDir(dirname):
        """
        清空目录指定目录，目录需要存在，否则报错。
        :param dirname: 目录名
        """
        if not os.path.exists(dirname):
            raise ValueError(dirname + " not exist")
        shutil.rmtree(dirname)
        os.mkdir(dirname)

    @staticmethod
    def getBasedirname(fullFilename):
        """
        根据文件全路径名获取基目录名并返回。
        :param fullFilename:
        :return:
        """
        return os.path.dirname(fullFilename)

    @staticmethod
    def getFilenameFromFullFilepathname(fullfilename):
        return os.path.basename(fullfilename)

    @staticmethod
    def stringMatched(str, patten):
        p = re.compile(patten)
        result = p.match(str)
        return result is not None

    @staticmethod
    def getPathFilenameList(dirname, patten = None):
        """
        获取目录下全部文件的文件路径名列表，不包含目录、链接等其他内容.
        也可以指定模式，搜索指定格式的文件名列表
        :param dirname:
        """
        if not FileNameUtil.fileExisted(dirname):
            raise ValueError("%s not existed"%dirname)

        result = [os.path.join(dirname, each) for each in os.listdir(dirname)]
        if patten is not None:
            result = [each for each in result if FileNameUtil.stringMatched(each,patten)]

        return result











def test():
    dirnameList = ["data","img"]
    dirname = FileNameUtil.getBasedirname(__file__)
    dirname = FileNameUtil.getDirname(dirname, dirnameList)
    pattern = r'.*\.jpg$'
    filenameList = FileNameUtil.getPathFilenameList(dirname, pattern)

    print(filenameList)
    print(FileNameUtil.getFilenameFromFullFilepathname(__file__))

if __name__ == "__main__":
    test()



