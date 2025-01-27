import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
import plotly.express as px
import dash
from dash import dcc, html
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import plotly.graph_objects as go
import numpy as np
from dash.dependencies import Input, Output
import plotly.figure_factory as ff

# Load and clean the data
data = pd.read_csv('insurance.csv')
data = data.dropna()
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

# Normalize numerical data
scaler = StandardScaler()
numerical_features = ['bmi', 'children']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Remove outliers based on smoking status
def remove_outliers(df, column, z_thresh=3):
    df = df.reset_index(drop=True)
    
    # Filter separately for both smokers and non-smokers 
    smokers_df = df[df['smoker_yes'] == True]
    non_smokers_df = df[df['smoker_yes'] == False]
    
    print(f"Before filtering:")
    print(f"Smokers: {len(smokers_df)}")
    print(f"Non-smokers: {len(non_smokers_df)}")
    
    # Calculate z-scores for smokers
    z_scores_smokers = np.abs((smokers_df[column] - smokers_df[column].mean()) / smokers_df[column].std())
    filtered_smokers = smokers_df[z_scores_smokers < z_thresh]
    
    # Calculate z-scores for non-smokers  
    z_scores_non = np.abs((non_smokers_df[column] - non_smokers_df[column].mean()) / non_smokers_df[column].std())
    filtered_non_smokers = non_smokers_df[z_scores_non < z_thresh]
    
    print(f"\nAfter filtering:")
    print(f"Smokers: {len(filtered_smokers)}")
    print(f"Non-smokers: {len(filtered_non_smokers)}")
    
    # Print stats about removed outliers
    print(f"\nOutliers removed:")
    print(f"Smokers: {len(smokers_df) - len(filtered_smokers)}")
    print(f"Non-smokers: {len(non_smokers_df) - len(filtered_non_smokers)}")
    
    # Combine filtered data
    return pd.concat([filtered_smokers, filtered_non_smokers]).reset_index(drop=True)

data = remove_outliers(data, 'charges')

# Continue with data processing and model training
X = data.drop('smoker_yes', axis=1)
y = data['smoker_yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

accuracy = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
precision = cross_val_score(model, X, y, cv=kf, scoring='precision')
f1 = cross_val_score(model, X, y, cv=kf, scoring='f1')

print(f'Random Forest - Accuracy: {accuracy.mean():.4f}')
print(f'Random Forest - Precision: {precision.mean():.4f}')
print(f'Random Forest - F1 Score: {f1.mean():.4f}')

# After model training, add these visualizations
def plot_roc_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return go.Figure(data=go.Scatter(x=fpr, y=tpr, 
                                    name=f'ROC curve (AUC = {roc_auc:.2f})'),
                    layout=go.Layout(title='ROC Curve',
                                   xaxis=dict(title='False Positive Rate'),
                                   yaxis=dict(title='True Positive Rate')))

def plot_pr_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:,1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    return go.Figure(data=go.Scatter(x=recall, y=precision,
                                    name='Precision-Recall curve'),
                    layout=go.Layout(title='Precision-Recall Curve',
                                   xaxis=dict(title='Recall'),
                                   yaxis=dict(title='Precision')))

def plot_learning_curves(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    
    return go.Figure(data=[
        go.Scatter(x=train_sizes, y=train_mean, name='Training score'),
        go.Scatter(x=train_sizes, y=val_mean, name='Cross-validation score')
    ], layout=go.Layout(title='Learning Curves',
                       xaxis=dict(title='Training Examples'),
                       yaxis=dict(title='Score')))

def plot_feature_importance(model, X):
    importances = model.feature_importances_
    features = X.columns
    
    return go.Figure(data=[
        go.Bar(x=features, y=importances)
    ], layout=go.Layout(title='Feature Importance',
                       xaxis=dict(title='Features'),
                       yaxis=dict(title='Importance')))

def plot_premium_analysis(data):
    return go.Figure(data=[
        go.Scatter(
            x=data['bmi'],
            y=data['charges'],
            mode='markers',
            marker=dict(
                size=data['age']/2,
                color=data['smoker_yes'],
                colorscale='Viridis',
            ),
            text=data['age'],
            name='Premium vs Risk Factors'
        )
    ], layout=go.Layout(
        title='Premium Analysis by Risk Factors',
        xaxis_title='BMI (Risk Factor)',
        yaxis_title='Charges ($)',
        showlegend=True
    ))

def plot_health_risk_distribution(data):
    age_groups = pd.qcut(data['age'], q=4, labels=['18-30', '31-45', '46-60', '60+'])
    risk_data = pd.DataFrame({
        'Age Group': age_groups,
        'BMI': data['bmi'],
        'Smoker': data['smoker_yes'],
        'Charges': data['charges']
    })
    
    return px.scatter(risk_data, 
                     x='BMI', 
                     y='Charges',
                     color='Age Group',
                     size='Charges',
                     facet_col='Smoker',
                     title='Health Risk Distribution')

# Setup Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Insurance Analytics Dashboard"),
    
    # Business Insights Section
    html.Div([
        html.H2("Business Insights"),
        html.Div([
            html.H3("Premium Optimization"),
            html.P("Analyze risk factors to optimize premium pricing"),
            dcc.Graph(id='premium-analysis', 
                     figure=plot_premium_analysis(data))
        ]),
        html.Div([
            html.H3("Health Risk Analysis"),
            html.P("Distribution of health risks across demographics"),
            dcc.Graph(id='health-risks',
                     figure=plot_health_risk_distribution(data))
        ])
    ]),
    
    # Scenario Analysis Section
    html.Div([
        html.H2("Scenario Analysis"),
        html.Div([
            html.Label("Select Risk Factor:"),
            dcc.Dropdown(
                id='risk-factor',
                options=[
                    {'label': 'Age', 'value': 'age'},
                    {'label': 'BMI', 'value': 'bmi'},
                    {'label': 'Smoking Status', 'value': 'smoker_yes'}
                ],
                value='bmi'
            )
        ]),
        dcc.Graph(id='scenario-impact')
    ]),
    
    # Existing visualizations
    html.H2("Model Performance"),
    dcc.Graph(id='roc-curve', figure=plot_roc_curve(model, X_test, y_test)),
    dcc.Graph(id='pr-curve', figure=plot_pr_curve(model, X_test, y_test)),
    dcc.Graph(id='learning-curves', figure=plot_learning_curves(model, X, y)),
    dcc.Graph(id='feature-importance', figure=plot_feature_importance(model, X))
])

# Add callback for scenario analysis
@app.callback(
    Output('scenario-impact', 'figure'),
    Input('risk-factor', 'value')
)
def update_scenario_impact(risk_factor):
    fig = px.scatter(data, 
                    x=risk_factor, 
                    y='charges',
                    color='smoker_yes',
                    trendline="ols",
                    title=f'Impact of {risk_factor} on Insurance Charges')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
