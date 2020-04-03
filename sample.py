class DataSample(object):
    def __init__(self, cid, content, label=None):
        super().__init__(self)
        self.cid = cid 
        self.content = content
        self.label = label
