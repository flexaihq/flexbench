import plotly.express as px


def create_color_mapping(accelerators: list[str]) -> dict[str, str]:
    acc_color_mapping = {}
    for idx, accelerator in enumerate(accelerators):
        acc_color_mapping[accelerator] = px.colors.qualitative.Plotly[
            idx % len(px.colors.qualitative.Plotly)
        ]
    return acc_color_mapping
