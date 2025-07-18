# FlexBoard

FlexBoard is a web-based dashboard for visualizing and analyzing performance data from MLPerf and flexbench benchmarks. It provides an interactive interface to explore various metrics, compare systems, and gain insights into the performance characteristics of different hardware and software configurations.

DISCLAIMER: This project is in early development and the integration with flexbench is not yet complete.

## Installation

```bash
cd flexboard
uv venv
source .venv/bin/activate
uv pip install -e .
python -m streamlit run app.py
```

## Next steps

- Proper integration with flexbench
- Add price for all accelerators (default is 1.0 USD/hr)
- Merge accelerators with different names (e.g. `AMD Instinct MI300X-NPS1-SPX-192GB-750W` and `AMD MI300X-NPS1-SPX-192GB-750W`)
- Add more plots and visualizations

## License and Copyright

This project is licensed under the [Apache License 2.0](LICENSE.md).

© 2025 FlexAI

## Authors and maintaners

[Daniel Altunay](https://www.linkedin.com/in/daltunay) and [Grigori Fursin](https://cKnowledge.org/gfursin) (FCS Labs)
