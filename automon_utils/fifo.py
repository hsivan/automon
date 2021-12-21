import numpy as np


class Fifo:
    """
    Used by the NodeStream class to maintain the sliding windows of the nodes efficiently.
    """
    
    def __init__(self, max_elements, num_attributes=1):
        self.max_elements = max_elements
        self.num_attributes = num_attributes
        self.__reset()

    def copy(self):
        f = Fifo(self.max_elements, self.num_attributes)
        f.fifo = self.fifo.copy()
        f.write_ptr = self.write_ptr
        f.num_elements = self.num_elements
        return f

    def __reset(self):
        self.fifo = np.zeros((self.max_elements, self.num_attributes))
        self.num_elements = 0
        self.write_ptr = 0  # Next element to override

    def reset(self):
        self.__reset()

    def is_window_full(self):
        return self.num_elements == self.max_elements

    def add_element(self, element):
        self.fifo[self.write_ptr] = element
        self.write_ptr = (self.write_ptr + 1) % self.max_elements
        if self.num_elements < self.max_elements:
            self.num_elements += 1

    def get_all_elements_ordered(self, b_squeeze=True):
        # Keep the order of the returned array
        if self.num_elements == self.max_elements:
            sorted_fifo = np.concatenate((self.fifo[self.write_ptr:self.max_elements], self.fifo[0:self.write_ptr]), axis=0)
            if b_squeeze:
                return np.squeeze(sorted_fifo)
            return sorted_fifo
        # Else, num_elements < max_elements
        if b_squeeze:
            return np.squeeze(self.fifo[:self.num_elements])
        return self.fifo[:self.num_elements]
    
    def get_all_elements_unordered(self, b_squeeze=True):
        if b_squeeze:
            return np.squeeze(self.fifo[:self.num_elements])
        return self.fifo[:self.num_elements]

    def get_oldest_element(self):
        if self.num_elements == 0:
            return None
        if self.num_elements < self.max_elements:
            return self.fifo[0]
        # If got here then elements are overridden, the oldest element is the one that is pointed by self.write_ptr
        return self.fifo[self.write_ptr]
    
    def get_num_element(self):
        return self.num_elements
    
    def print_fifo(self):
        print("=============")
        print(self.fifo)
        print("write_ptr", self.write_ptr)
        print("=============")


if __name__ == "__main__":    
    fifo = Fifo(4, 3)
    print(fifo.get_oldest_element())
    fifo.add_element(np.ones(3)*1)
    fifo.add_element(np.ones(3)*2)
    print(fifo.get_oldest_element())
    fifo.print_fifo()
    print("All elements", fifo.get_all_elements_unordered())
    fifo.add_element(np.ones(3)*3)
    fifo.add_element(np.ones(3)*4)
    fifo.add_element(np.ones(3)*5)
    fifo.add_element(np.ones(3)*6)
    fifo.add_element(np.ones(3)*7)
    fifo.print_fifo()
    print("Oldest element", fifo.get_oldest_element())
    print("All elements", fifo.get_all_elements_unordered())
    
    fifo = Fifo(2, 1)
    fifo.print_fifo()
    print("Oldest element", fifo.get_oldest_element())
    print("All elements", fifo.get_all_elements_unordered())
