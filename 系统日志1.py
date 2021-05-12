import mmap
import contextlib
import Evtx.Evtx
from Evtx.Evtx import FileHeader
from Evtx.Views import evtx_file_xml_view
from xml.dom import minidom


def MyFun():
    EvtxPath = "./System.evtx"

    with open(EvtxPath, 'r') as f:
        with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as buf:
            fh = FileHeader(buf, 0)
            for xml, record in evtx_file_xml_view(fh):
                # ֻ����¼�IDΪx������
                InterestEvent(xml, 20001)
            print("")
# ���˵�����Ҫ���¼����������Ȥ���¼�


def InterestEvent(xml, EventID):
    xmldoc = minidom.parseString(xml)
    eventid = xmldoc.getElementsByTagName(EventID)
    print(eventid)
    # ��ȡEventID�ڵ���¼�ID
    booknode = xmldoc.getElementsByTagName(EventID)
    for booklist in booknode:
        bookdict = {}
        eventid = booklist.getAttribute(EventID)
        print(eventid)

    if eventid == EventID:
        print(xml)


if __name__ == '__main__':
    MyFun()
