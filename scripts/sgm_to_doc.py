import sys
from bs4 import BeautifulSoup

with open (sys.argv[1], 'r') as f:
    soup = BeautifulSoup(f.read(), "lxml")
    docs = soup.find_all("doc")
    for doc in docs:
        docid = doc['docid']
        for i in range(len(doc.find_all("seg"))):
            print(docid)
