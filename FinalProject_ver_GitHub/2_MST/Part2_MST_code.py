import os
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt

# Set the file path
file_directory = r'2_MST'

# English-translated industry names for legend
english_labels = {
    0: 'Transportation_Bond', 1: 'Media_Bond', 2: 'Utilities_Bond', 3: 'Agriculture_Bond', 4: 'Pharmaceuticals_Bond',
    5: 'Retail_Bond', 6: 'Defense_Bond', 7: 'Chemicals_Bond', 8: 'Home_Appliances_Bond', 9: 'Building_Materials_Bond',
    10: 'Construction_Decoration_Bond', 11: 'Real_Estate_Bond', 12: 'Non_Ferrous_Metals_Bond', 13: 'Machinery_Bond',
    14: 'Automotive_Bond', 15: 'Coal_Bond', 16: 'Environmental_Protection_Bond', 17: 'Electric_Power_Equipment_Bond',
    18: 'Electronics_Bond', 19: 'Petrochemicals_Bond', 20: 'Social_Services_Bond', 21: 'Textiles_Bond',
    22: 'Comprehensive_Bond',
    23: 'Computers_Bond', 24: 'Light_Industry_Bond', 25: 'Telecommunications_Bond', 26: 'Steel_Bond',
    27: 'Non_Bank_Finance_Bond',
    28: 'Food_Beverage_Bond', 29: 'Transportation_Stock', 30: 'Media_Stock', 31: 'Utilities_Stock',
    32: 'Agriculture_Stock',
    33: 'Pharmaceuticals_Stock', 34: 'Retail_Stock', 35: 'Defense_Stock', 36: 'Chemicals_Stock',
    37: 'Home_Appliances_Stock',
    38: 'Building_Materials_Stock', 39: 'Construction_Decoration_Stock', 40: 'Real_Estate_Stock',
    41: 'Non_Ferrous_Metals_Stock',
    42: 'Machinery_Stock', 43: 'Automotive_Stock', 44: 'Coal_Stock', 45: 'Environmental_Protection_Stock',
    46: 'Electric_Power_Equipment_Stock',
    47: 'Electronics_Stock', 48: 'Petrochemicals_Stock', 49: 'Social_Services_Stock', 50: 'Textiles_Stock',
    51: 'Comprehensive_Stock',
    52: 'Computers_Stock', 53: 'Light_Industry_Stock', 54: 'Telecommunications_Stock', 55: 'Steel_Stock',
    56: 'Non_Bank_Finance_Stock',
    57: 'Food_Beverage_Stock'
}

# Loop through all npy files in the folder
for file_name in os.listdir(file_directory):
    if file_name.endswith('.npy'):
        # Load the npy file
        file_path = os.path.join(file_directory, file_name)
        data = np.load(file_path)

        # Generate the minimum spanning tree
        mst_matrix = minimum_spanning_tree(data).toarray()

        # Create a graph structure
        graph = nx.from_numpy_array(mst_matrix)

        # Set node labels as node numbers
        node_labels = {i: str(i) for i in range(data.shape[0])}

        # Draw the graph
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)  # High resolution
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, ax=ax, with_labels=True, labels=node_labels, node_size=500, node_color='lightblue',
                font_size=10, edge_color='gray')

        # Save the image as PNG with a transparent background
        save_path = os.path.join(file_directory, f'MST_{file_name.split(".")[0]}.png')
        plt.savefig(save_path, format='png', bbox_inches='tight', transparent=True)
        plt.close()

        print(f'Saved MST for {file_name} as {save_path}')

        # Generate the correspondence between node numbers and labels as text
        correspondence_text = "\n".join([f'Node {i}: {english_labels[i]}' for i in range(data.shape[0])])

        # Save the correspondence text to a file
        text_save_path = os.path.join(file_directory, f'label_{file_name.split(".")[0]}.txt')
        with open(text_save_path, 'w') as f:
            f.write(correspondence_text)

        print(f'Saved labels correspondence for {file_name} as {text_save_path}')
