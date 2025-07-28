import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Hsiao's Homogeneity Test", layout="wide")

st.title("Hsiao's Homogeneity Test in Panel Data")
st.write("Created by: Dr. Merwan Roudane")


# Function to calculate F-statistic
def calculate_f_test(restricted_model, unrestricted_model):
    """Calculate F-statistic for model comparison similar to R's anova function"""
    rss_r = sum(restricted_model.resid ** 2)
    rss_ur = sum(unrestricted_model.resid ** 2)
    df_r = restricted_model.df_resid
    df_ur = unrestricted_model.df_resid

    # Calculate F-statistic
    df_diff = df_r - df_ur
    f_stat = ((rss_r - rss_ur) / df_diff) / (rss_ur / df_ur)

    # Calculate p-value
    p_value = 1 - stats.f.cdf(f_stat, df_diff, df_ur)

    return {
        "RSS Restricted": rss_r,
        "RSS Unrestricted": rss_ur,
        "DF Restricted": df_r,
        "DF Unrestricted": df_ur,
        "DF Difference": df_diff,
        "F-statistic": f_stat,
        "p-value": p_value
    }


# Function to format model summary with HTML
def format_model_summary(model):
    """Format model summary table with HTML for better visualization"""
    results = model.summary()

    # Convert to HTML
    results_html = results.tables[1].as_html()

    # Add custom styling
    styled_html = f"""
    <style>
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
        }}
        .summary-table th, .summary-table td {{
            padding: 8px;
            text-align: right;
            border-bottom: 1px solid #ddd;
        }}
        .summary-table th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .significant {{
            font-weight: bold;
            color: #2470dc;
        }}
        .highly-significant {{
            font-weight: bold;
            color: #d62728;
        }}
    </style>

    <h3>Model Summary</h3>
    <p>R-squared: <b>{model.rsquared:.4f}</b> | Adjusted R-squared: <b>{model.rsquared_adj:.4f}</b></p>
    <p>F-statistic: <b>{model.fvalue:.4f}</b> | Prob (F-statistic): <b>{model.f_pvalue:.4f}</b></p>
    <p>AIC: <b>{model.aic:.2f}</b> | BIC: <b>{model.bic:.2f}</b></p>

    {results_html.replace('<table', '<table class="summary-table"')}

    <script>
        document.querySelectorAll('.summary-table tr').forEach(row => {{
            const pValueCell = row.cells[row.cells.length - 1];
            if (pValueCell && !isNaN(parseFloat(pValueCell.textContent))) {{
                const pValue = parseFloat(pValueCell.textContent);
                if (pValue < 0.01) {{
                    row.querySelectorAll('td').forEach(cell => {{
                        cell.classList.add('highly-significant');
                    }});
                }} else if (pValue < 0.05) {{
                    row.querySelectorAll('td').forEach(cell => {{
                        cell.classList.add('significant');
                    }});
                }}
            }}
        }});
    </script>
    """

    return styled_html


# File uploader
uploaded_file = st.file_uploader("Choose an Excel File", type=["xlsx"])

if uploaded_file is not None:
    # Read the data
    try:
        df = pd.read_excel(uploaded_file)
        st.success(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Display the data
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Variable selection
    col1, col2 = st.columns(2)

    with col1:
        # Get column names
        columns = df.columns.tolist()

        # Select variables
        id_var = st.selectbox("Select Cross-sectional ID variable:", columns)
        time_var = st.selectbox("Select Time variable:", columns)

    with col2:
        dependent_var = st.selectbox("Select dependent variable:", columns)
        independent_vars = st.multiselect("Select independent variables:", columns)

    # Information box
    with st.expander("About Hsiao's Homogeneity Test"):
        st.markdown("""
        ### Theoretical Background

        Hsiao's homogeneity test is a sequence of F-tests used to determine the appropriate panel data model specification.

        #### Mathematical Formulation
        """)

        st.latex(r'''
        \text{The general model:} \quad y_{it} = \alpha_i + \sum_{k=1}^{K} \beta_{ki} x_{kit} + \varepsilon_{it}
        ''')

        st.markdown("""
        Where:
        - $y_{it}$ is the dependent variable for unit $i$ at time $t$
        - $\\alpha_i$ is the intercept for unit $i$
        - $\\beta_{ki}$ is the slope coefficient for the $k$-th independent variable for unit $i$
        - $x_{kit}$ is the $k$-th independent variable for unit $i$ at time $t$
        - $\\varepsilon_{it}$ is the error term

        The tests evaluate:
        1. **Homogeneity of slopes and intercepts**: $H_0: \\alpha_1 = \\alpha_2 = ... = \\alpha_N$ and $\\beta_{k1} = \\beta_{k2} = ... = \\beta_{kN}$ for all $k$
        2. **Homogeneity of slopes**: $H_0: \\beta_{k1} = \\beta_{k2} = ... = \\beta_{kN}$ for all $k$ (allowing intercepts to vary)
        3. **Homogeneity of intercepts**: $H_0: \\alpha_1 = \\alpha_2 = ... = \\alpha_N$ (assuming slopes are constant)

        Reject the null hypothesis if p-value < 0.05.
        """)

    # Run tests button
    run_tests = st.button("Run Tests and Visualizations")

    if run_tests:
        if not independent_vars:
            st.error("Please select at least one independent variable.")
            st.stop()

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Preparing data...")

        # Convert ID to factor/category
        df[id_var] = df[id_var].astype('category')

        # Create formula strings for the models
        independent_str = " + ".join(independent_vars)

        # Full model with interaction terms
        interaction_terms = " + ".join([f"C({id_var}):{var}" for var in independent_vars])
        full_formula = f"{dependent_var} ~ {independent_str} + C({id_var}) + {interaction_terms}"

        # Reduced models
        reduced_formula_1 = f"{dependent_var} ~ {independent_str}"
        reduced_formula_2 = f"{dependent_var} ~ {independent_str} + C({id_var})"
        reduced_formula_3 = f"{dependent_var} ~ {independent_str}"

        # Fit models
        status_text.text("Fitting models...")
        progress_bar.progress(20)

        try:
            model_full = ols(full_formula, data=df).fit()
            model_reduced_1 = ols(reduced_formula_1, data=df).fit()
            model_reduced_2 = ols(reduced_formula_2, data=df).fit()
            model_reduced_3 = ols(reduced_formula_3, data=df).fit()
        except Exception as e:
            st.error(f"Error fitting models: {e}")
            st.error("This might be due to perfect multicollinearity or missing values.")
            st.stop()

        status_text.text("Calculating test statistics...")
        progress_bar.progress(40)

        # Calculate F-statistics
        f_test1 = calculate_f_test(model_reduced_1, model_full)
        f_test2 = calculate_f_test(model_reduced_2, model_full)
        f_test3 = calculate_f_test(model_reduced_3, model_reduced_2)

        status_text.text("Displaying results...")
        progress_bar.progress(60)

        # Display results
        st.header("Hsiao's Homogeneity Tests")

        # Create a styled table for test results
        test_results = pd.DataFrame({
            "Test": ["Homogeneity of slopes and intercepts", "Homogeneity of slopes", "Homogeneity of intercepts"],
            "F-statistic": [f"{f_test1['F-statistic']:.4f}", f"{f_test2['F-statistic']:.4f}",
                            f"{f_test3['F-statistic']:.4f}"],
            "p-value": [f"{f_test1['p-value']:.4f}", f"{f_test2['p-value']:.4f}", f"{f_test3['p-value']:.4f}"],
            "Degrees of freedom": [f"{f_test1['DF Difference']} and {f_test1['DF Unrestricted']}",
                                   f"{f_test2['DF Difference']} and {f_test2['DF Unrestricted']}",
                                   f"{f_test3['DF Difference']} and {f_test3['DF Unrestricted']}"],
            "Result": [
                "Reject null hypothesis" if f_test1['p-value'] < 0.05 else "Fail to reject null hypothesis",
                "Reject null hypothesis" if f_test2['p-value'] < 0.05 else "Fail to reject null hypothesis",
                "Reject null hypothesis" if f_test3['p-value'] < 0.05 else "Fail to reject null hypothesis"
            ]
        })


        # Apply styling to the dataframe
        def highlight_significant(val):
            if "Reject" in val:
                return 'background-color: rgba(255, 0, 0, 0.2); font-weight: bold'
            else:
                return 'background-color: rgba(0, 255, 0, 0.2)'


        styled_results = test_results.style.applymap(highlight_significant, subset=['Result'])

        st.table(styled_results)

        # Explanation of test results
        st.subheader("Interpretation of Test Results")

        st.markdown("""
        ### Test 1: Homogeneity of slopes and intercepts
        """)
        st.latex(r'''
        H_0: \alpha_1 = \alpha_2 = ... = \alpha_N \text{ and } \beta_{k1} = \beta_{k2} = ... = \beta_{kN} \text{ for all } k
        ''')
        if f_test1['p-value'] < 0.05:
            st.markdown(
                "**Result**: Reject null hypothesis - The model coefficients vary across cross-sections. This suggests heterogeneity in the panel data.")
        else:
            st.markdown(
                "**Result**: Fail to reject null hypothesis - The model coefficients are constant across cross-sections. This suggests a pooled OLS model may be appropriate.")

        st.markdown("""
        ### Test 2: Homogeneity of slopes
        """)
        st.latex(r'''
        H_0: \beta_{k1} = \beta_{k2} = ... = \beta_{kN} \text{ for all } k
        ''')
        if f_test2['p-value'] < 0.05:
            st.markdown(
                "**Result**: Reject null hypothesis - The slopes vary across cross-sections. This suggests different relationships between independent and dependent variables across units.")
        else:
            st.markdown(
                "**Result**: Fail to reject null hypothesis - The slopes are constant across cross-sections. This suggests a fixed or random effects model may be appropriate.")

        st.markdown("""
        ### Test 3: Homogeneity of intercepts
        """)
        st.latex(r'''
        H_0: \alpha_1 = \alpha_2 = ... = \alpha_N
        ''')
        if f_test3['p-value'] < 0.05:
            st.markdown(
                "**Result**: Reject null hypothesis - The intercepts vary across cross-sections. This suggests fixed effects may be present.")
        else:
            st.markdown(
                "**Result**: Fail to reject null hypothesis - The intercepts are constant across cross-sections. This suggests a pooled OLS model may be appropriate.")

        # Model selection recommendation
        st.subheader("Model Selection Recommendation")

        model_rec = ""
        if f_test1['p-value'] >= 0.05:
            model_rec = "✅ **Pooled OLS Model**: Use a simple pooled model with common slopes and intercepts."
            st.info(model_rec)
        elif f_test2['p-value'] >= 0.05 and f_test3['p-value'] < 0.05:
            model_rec = "✅ **Fixed Effects Model**: Use a model with fixed effects (different intercepts but common slopes)."
            st.info(model_rec)
        elif f_test2['p-value'] < 0.05:
            model_rec = "✅ **Random Coefficients Model**: Consider a model that allows both slopes and intercepts to vary across cross-sections."
            st.info(model_rec)

        # Display model summary in an expander
        with st.expander("Full Model Summary"):
            st.components.v1.html(format_model_summary(model_full), height=500, scrolling=True)

        status_text.text("Creating visualizations...")
        progress_bar.progress(80)

        # Visualizations
        st.header("Visualizations")

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Time Series",
            "Correlation Analysis",
            "Distribution Analysis",
            "Residual Analysis"
        ])

        # 1. Time series plot with Plotly
        with tab1:
            st.subheader("Time Series Plot")

            # Create a Plotly figure
            fig = px.line(df, x=time_var, y=dependent_var, color=id_var, markers=True,
                          title=f"{dependent_var} over Time by {id_var}")

            fig.update_layout(
                xaxis_title=time_var,
                yaxis_title=dependent_var,
                legend_title=id_var,
                hovermode="closest"
            )

            if len(df[id_var].unique()) > 15:
                fig.update_layout(showlegend=False)
                st.write("Note: Legend hidden due to large number of cross-sections")

            st.plotly_chart(fig, use_container_width=True)

            # Theoretical explanation
            st.markdown("""
            ### Time Series Analysis in Panel Data

            The time series plot shows how the dependent variable evolves over time for each cross-sectional unit. 
            Parallel trends between units suggest homogeneity, while diverging trends suggest heterogeneity.

            Key patterns to look for:
            - **Parallel lines**: Suggest constant slopes across units (homogeneous slopes)
            - **Different intercepts**: Suggest unit-specific fixed effects
            - **Crossing lines**: Suggest heterogeneous slopes across units
            """)

        # 2. Correlation analysis
        with tab2:
            st.subheader("Correlation Analysis")

            # Create correlation matrix
            selected_cols = [dependent_var] + independent_vars
            corr_matrix = df[selected_cols].corr()

            # Plotly heatmap
            fig = px.imshow(corr_matrix,
                            text_auto=True,
                            color_continuous_scale="RdBu_r",
                            title="Correlation Matrix",
                            zmin=-1, zmax=1)

            st.plotly_chart(fig, use_container_width=True)

            # Scatter plots with regression lines
            st.subheader("Scatter Plots with Regression Lines")

            for var in independent_vars:
                # Create scatter plot by group
                fig = px.scatter(df, x=var, y=dependent_var, color=id_var,
                                 opacity=0.7, trendline="ols",
                                 title=f"{dependent_var} vs {var}")

                fig.update_layout(
                    xaxis_title=var,
                    yaxis_title=dependent_var
                )

                if len(df[id_var].unique()) > 10:
                    fig.update_layout(showlegend=False)
                    st.write(f"Note: Legend hidden due to large number of {id_var} categories")

                st.plotly_chart(fig, use_container_width=True)

            # Theoretical explanation
            st.markdown("""
            ### Correlation Analysis in Panel Data

            The correlation matrix and scatter plots provide insights into the relationships between variables.

            Key insights:
            - **High correlation** between independent variables may indicate multicollinearity
            - **Different regression slopes** across groups suggest heterogeneous relationships
            - **Clustering** of points by group suggests group-specific effects

            The relationship between variables can be mathematically represented as:
            """)

            st.latex(r'''
            \hat{y}_{it} = \hat{\alpha}_i + \hat{\beta}_{1i}x_{1it} + \hat{\beta}_{2i}x_{2it} + ... + \hat{\beta}_{ki}x_{kit}
            ''')

        # 3. Distribution analysis
        with tab3:
            st.subheader("Distribution Analysis")

            # Determine units to plot
            unique_ids = df[id_var].unique()
            if len(unique_ids) > 15:
                st.write("Note: Too many cross-sections. Showing first 15 only.")
                ids_to_plot = unique_ids[:15]
                plot_df = df[df[id_var].isin(ids_to_plot)]
            else:
                plot_df = df

            # Box plot with Plotly
            fig = px.box(plot_df, x=id_var, y=dependent_var,
                         title=f"Distribution of {dependent_var} by {id_var}")

            fig.update_layout(
                xaxis_title=id_var,
                yaxis_title=dependent_var
            )

            st.plotly_chart(fig, use_container_width=True)

            # Violin plot with Plotly
            fig = px.violin(plot_df, x=id_var, y=dependent_var, box=True, points="all",
                            title=f"Violin Plot of {dependent_var} by {id_var}")

            fig.update_layout(
                xaxis_title=id_var,
                yaxis_title=dependent_var
            )

            st.plotly_chart(fig, use_container_width=True)

            # Density plot for dependent variable
            fig = px.histogram(df, x=dependent_var, color=id_var, marginal="rug",
                               title=f"Distribution of {dependent_var}",
                               opacity=0.7, histnorm="probability density")

            fig.update_layout(
                xaxis_title=dependent_var,
                yaxis_title="Density"
            )

            if len(df[id_var].unique()) > 10:
                fig.update_layout(showlegend=False)
                st.write(f"Note: Legend hidden due to large number of {id_var} categories")

            st.plotly_chart(fig, use_container_width=True)

            # Theoretical explanation
            st.markdown("""
            ### Distribution Analysis in Panel Data

            Box plots, violin plots, and density plots provide insights into the distribution of the dependent variable across cross-sectional units.

            Key insights:
            - **Different medians** across units suggest unit-specific intercepts (fixed effects)
            - **Different spreads** suggest heteroscedasticity or unit-specific variance
            - **Different shapes** suggest fundamentally different distributions across units

            These differences can inform model specification, particularly whether to include unit-specific effects.
            """)

        # 4. Residual analysis
        with tab4:
            st.subheader("Residual Analysis")

            # Add fitted values and residuals to dataframe
            df_model = df.copy()
            df_model['fitted'] = model_full.fittedvalues
            df_model['residuals'] = model_full.resid

            # Residual vs Fitted plot
            fig = px.scatter(df_model, x='fitted', y='residuals', color=id_var,
                             title="Residuals vs Fitted Values")

            fig.add_shape(
                type="line",
                x0=min(df_model['fitted']),
                y0=0,
                x1=max(df_model['fitted']),
                y1=0,
                line=dict(color="red", width=2, dash="dash")
            )

            fig.update_layout(
                xaxis_title="Fitted values",
                yaxis_title="Residuals"
            )

            if len(df[id_var].unique()) > 10:
                fig.update_layout(showlegend=False)

            st.plotly_chart(fig, use_container_width=True)

            # QQ plot using Plotly
            fig = go.Figure()

            qq_data = stats.probplot(model_full.resid, dist="norm")
            x = np.array([min(qq_data[0][0]), max(qq_data[0][0])])

            fig.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Residuals'
            ))

            fig.add_trace(go.Scatter(
                x=x,
                y=qq_data[1][0] + qq_data[1][1] * x,
                mode='lines',
                name='Theoretical Line',
                line=dict(color='red', dash='dash')
            ))

            fig.update_layout(
                title="Normal Q-Q Plot",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Histogram of residuals
            fig = px.histogram(df_model, x='residuals', nbins=30,
                               title="Histogram of Residuals",
                               marginal="rug")

            fig.update_layout(
                xaxis_title="Residuals",
                yaxis_title="Frequency"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Residuals by cross-section
            fig = px.box(df_model, x=id_var, y='residuals',
                         title="Residuals by Cross-section")

            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=0,
                x1=len(df[id_var].unique()) - 0.5,
                y1=0,
                line=dict(color="red", width=2, dash="dash")
            )

            fig.update_layout(
                xaxis_title=id_var,
                yaxis_title="Residuals"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Theoretical explanation
            st.markdown("""
            ### Residual Analysis in Panel Data

            Residual analysis helps assess the validity of model assumptions.

            Key diagnostics:
            - **Residuals vs Fitted**: Check for non-linearity and heteroscedasticity
            - **Q-Q Plot**: Check for normality of residuals
            - **Histogram**: Check for symmetry and outliers
            - **Residuals by Cross-section**: Check for systematic differences across units

            The residuals represent the unexplained variation in the model:
            """)

            st.latex(r'''
            \hat{\varepsilon}_{it} = y_{it} - \hat{y}_{it} = y_{it} - (\hat{\alpha}_i + \sum_{k=1}^{K} \hat{\beta}_{ki} x_{kit})
            ''')

        # Complete the progress bar
        progress_bar.progress(100)
        status_text.text("Analysis complete!")