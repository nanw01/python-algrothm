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
            print(evtx_file_xml_view(fh))
            for xml, record in evtx_file_xml_view(fh):
                InterestEvent(xml, 20001)
            print("123")


def InterestEvent(xml, EventID):
    xmldoc = minidom.parseString(xml)
    eventid = xmldoc.getElementsByTagName('EventID')
    print('333', eventid)

    booknode = xmldoc.getElementsByTagName('Event')
    for booklist in booknode:
        bookdict = {}
        print('888', booklist)
        eventid = booklist.getAttribute('EventID')
        print('444', eventid)

    if eventid == EventID:
        print('777', xml)


if __name__ == '__main__':
    MyFun()
