## Installation

1. Create and activate a new conda environment:

   ```bash
   conda create -n janusv python=3.10 -y
   conda activate janusv
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Flash-Att Installation

To install **flash-attn**, use the following command:

```bash
pip install flash-attn --no-build-isolation
```

### Troubleshooting

If the above command raises an error, please install the up-to-date version of **flash-attn** from the official release page:

- Official release page: [https://github.com/Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases)

**Important**: Make sure to select the version with `abiFALSE` rather than `abiTrue`.

You can install the correct version with this command:

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

For more details, refer to the [issue discussion on GitHub](https://github.com/Dao-AILab/flash-attention/issues/224#issuecomment-2084366991).
```