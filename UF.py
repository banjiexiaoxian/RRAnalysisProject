class UF:
    ##elements [[corepoint_index_in_elements,corepoint_index_for_label]..[]]
    def __init__(self, elements):
        self.elements = elements
        self.cluster_no = [i for i in range(len(elements))]
        self.component_size = [1 for i in range(len(elements))]
    ##此处有问题！cluster_no代表的标号不一定是cluster号码，cluster必须是根节点
    def union(self, p, q):  # initialize N sites with integer names
        p_cluster_no = self.find(p)
        q_cluster_no = self.find(q)
        if p_cluster_no != q_cluster_no:
            if self.component_size[p_cluster_no] > self.component_size[q_cluster_no]:
                self.cluster_no[q_cluster_no] = p_cluster_no
                self.component_size[p_cluster_no] += self.component_size[q_cluster_no]
                self.component_size[q_cluster_no] = 0
            else:
                self.cluster_no[p_cluster_no] = q_cluster_no
                self.component_size[q_cluster_no] += self.component_size[p_cluster_no]
                self.component_size[p_cluster_no] = 0
        return

    def find(self, p):  # return component identifier for p
        p_cluster_no = self.cluster_no[p]
        while(p_cluster_no != p):
            p = p_cluster_no
            p_cluster_no = self.cluster_no[p_cluster_no]
        return p

    def connected(self, p, q):  # return true if p and q are in the same component
        return self.find(p) == self.find(q)

    def assign_all_2_root(self):
        for element in self.elements:
            self.cluster_no[element[1]] = self.find(element[1])
        return

    def clustered_elements(self):
        self.assign_all_2_root()
        unique_labels = set(self.cluster_no)
        clustered_elements = dict()
        for element in self.elements:
            if self.cluster_no[element[1]] not in clustered_elements.keys():
                clustered_elements[self.cluster_no[element[1]]] = []
            clustered_elements[self.cluster_no[element[1]]].append(element[0])
        return clustered_elements

    def cluster_component_size(self):
        return self.component_size
##end