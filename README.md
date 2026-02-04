```markdown
# Global_happiness_analysis

本仓库包含基于世界幸福数据的分析脚本与交互式演示（Streamlit）。主要文件与说明：

目录结构建议
- data/                  -> 放置 `Happy_Updated.csv`
- R/                     -> R 脚本（data_analysis_happiness_modeling.R）
- figures/               -> 运行后生成的图像
- outputs/               -> 运行后生成的 CSV/摘要
- app/                   -> Streamlit 应用代码（app.py）
- README.md
- .gitignore

快速运行（R 脚本）
1. 将 `Happy_Updated.csv` 放入 `data/`
2. 在 R 中安装依赖（建议使用 renv）：
   install.packages(c("tidyverse","here","broom","patchwork","glmnet","caret","MASS","car","lmtest","corrplot"))
3. 运行 `R/data_analysis_happiness_modeling.R`
4. 查看 `outputs/`（model_evaluation_testset.csv、coefficients_*.csv、automated_model_summaries_cn.txt）和 `figures/` 下的图像

运行 Streamlit Web 应用
1. 进入 app/ 目录
2. 创建 Python 虚拟环境并安装依赖： `pip install -r requirements.txt`
3. 运行： `streamlit run app.py --server.port 8501`
4. 在浏览器打开 http://localhost:8501

提交到 GitHub
- 我提供了 commit / PR 模板与命令示例（参见仓库根目录的 .github/ 文件夹）
```