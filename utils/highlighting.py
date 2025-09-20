from IPython.display import display, HTML


def highlight_strings(strings, values, num_gen_tokens: int = 32):
    """
    Highlights strings with corresponding values using a gradient and displays them with colored backgrounds in a Jupyter Notebook.
    The last 32 entries are highlighted with a different text color (blue).

    Args:
        strings (list of str): List of strings.
        values (list of float): List of float values in the range [0, 1].

    Returns:
        None

    Raises:
        ValueError: If the lengths of the two lists do not match or values are not in [0, 1].
    """
    if len(strings) != len(values):
        raise ValueError("The lengths of strings and values lists must be equal.")

    if not all(0 <= v <= 1 for v in values):
        raise ValueError("All values in the second list must be in the range [0, 1].")

    def value_to_color(value):
        """Convert a float value to a gradient color from white to red."""
        red_intensity = int(value * 255)
        return (
            f"rgba(255, 0, 0,{value})" if value > 0.0 else "rgba(255, 255, 255, 255)"
        )

    # Display with gradient-colored backgrounds and special color for the last 32 entries
    styled_strings = []
    total_strings = len(strings)
    for i, (string, value) in enumerate(zip(strings, values)):
        string = string.replace("\n", "<br>")
        background_color = value_to_color(value)

        # Set a different text color for the last 32 entries
        if i >= total_strings - num_gen_tokens:
            text_color = "rgb(0, 0, 255)"  # Blue color for text
        else:
            text_color = "rgb(0, 0, 0)"  # Default text color (black)

        # Append styled string
        styled_strings.append(
            f'<span style="background-color: {background_color}; color: {text_color}; padding: 2px;">{string}</span>'
        )

    # Join styled strings for readable output
    styled_output = " ".join(styled_strings)
    display(
        HTML(f"""
                <div style="background-color: white; color: black; padding: 10px;">
                    {styled_output}
                </div>
            """)
    )
