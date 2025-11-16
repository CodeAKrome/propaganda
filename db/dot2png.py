#!/usr/bin/env python
import graphviz
import sys

# dotfile outfile_base

class gr:
    def __init__(self):
        self.text = "foo"

    def _create_graph(self, input_data):
        """
        Create a graphviz.Source object from input data.

        :param input_data: Either a file path, list of strings, or a single string
        :return: graphviz.Source object
        """
        if isinstance(input_data, str):
            # Check if it's a file path
            if input_data.endswith(".dot") or input_data.endswith(".gv"):
                with open(input_data, "r") as f:
                    dot_content = f.read()
            else:
                # Treat it as dot content string
                dot_content = input_data
        elif isinstance(input_data, list):
            # Join lines into a single string
            dot_content = "\n".join(input_data)
        else:
            raise ValueError(
                "Input must be a file path, list of strings, or a dot content string"
            )

        return graphviz.Source(dot_content)

    def render_to_file(self, input_data, output_file=None, format="png"):
        """
        Render a GraphViz image and save to a file.

        :param input_data: File path, list of strings, or dot content string
        :param output_file: Name of the output file (without extension)
        :param format: Output format (e.g., 'png', 'svg', 'pdf')
        :return: Path to the rendered image
        """
        self.graph = self._create_graph(input_data)
        if not output_file:
            output_file = "output"
        try:
            # Render the graph
            rendered_path = self.graph.render(
                filename=output_file, format=format, cleanup=True
            )
            print(f"Image rendered successfully: {rendered_path}")
            return rendered_path
        except graphviz.ExecutableNotFound:
            print("Error: Graphviz executable not found. Please install Graphviz.")
        except Exception as e:
            print(f"An error occurred while rendering the image: {str(e)}")


g = gr()
with open(sys.argv[1], "r") as f:
    data = f.read()

g.render_to_file(data, sys.argv[2])
