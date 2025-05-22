import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import os
import subprocess
from jinja2 import Template
from jinja2 import Environment

# Download NLTK data files (if not already downloaded)
nltk.download('stopwords')

def analyze_dataset(file_path, report_data):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Ensure the dataset contains the necessary columns
    if 'text' not in data.columns or 'label' not in data.columns:
        raise ValueError(f"The dataset at {file_path} must contain 'text' and 'label' columns.")

    # Check for missing values
    missing_text = data['text'].isnull().sum()
    missing_label = data['label'].isnull().sum()
    print(f"Missing 'text' entries: {missing_text}")
    print(f"Missing 'label' entries: {missing_label}")

    # Drop missing values
    data.dropna(subset=['text', 'label'], inplace=True)

    # Convert labels to string if they are not already
    data['label'] = data['label'].astype(str)

    # Number of samples (sentences)
    num_samples = len(data)

    # Sentence length (in terms of word count)
    data['sentence_length_words'] = data['text'].apply(lambda x: len(str(x).split()))

    # Sentence length (in terms of character count)
    data['sentence_length_chars'] = data['text'].apply(lambda x: len(str(x)))

    # Average, max, and min sentence length (words)
    avg_length_words = data['sentence_length_words'].mean()
    max_length_words = data['sentence_length_words'].max()
    min_length_words = data['sentence_length_words'].min()

    # Average, max, and min sentence length (chars)
    avg_length_chars = data['sentence_length_chars'].mean()
    max_length_chars = data['sentence_length_chars'].max()
    min_length_chars = data['sentence_length_chars'].min()

    # Label distribution
    label_distribution = data['label'].value_counts()
    label_proportion = data['label'].value_counts(normalize=True) * 100

    # Number of unique labels
    num_unique_labels = data['label'].nunique()

    # Number of unique sentences (in case of duplicates)
    num_unique_sentences = data['text'].nunique()

    # Vocabulary size
    all_words = ' '.join(data['text']).split()
    vocab_size = len(set(all_words))

    # Most common words (excluding stop words)
    stop_words = set(stopwords.words('english'))
    words_no_stop = [word for word in all_words if word.lower() not in stop_words]
    most_common_words = Counter(words_no_stop).most_common(20)

    # Additional statistics: standard deviation, median, etc.
    std_length_words = data['sentence_length_words'].std()
    median_length_words = data['sentence_length_words'].median()
    quantiles_words = data['sentence_length_words'].quantile([0.25, 0.5, 0.75])

    filename = os.path.basename(file_path)
    filename_no_ext = os.path.splitext(filename)[0]

    # Store results in report_data
    report_data[filename_no_ext] = {
        'num_samples': num_samples,
        'missing_text': missing_text,
        'missing_label': missing_label,
        'num_unique_sentences': num_unique_sentences,
        'num_unique_labels': num_unique_labels,
        'vocab_size': vocab_size,
        'avg_length_words': avg_length_words,
        'max_length_words': max_length_words,
        'min_length_words': min_length_words,
        'avg_length_chars': avg_length_chars,
        'max_length_chars': max_length_chars,
        'min_length_chars': min_length_chars,
        'label_distribution': label_distribution.to_dict(),
        'label_proportion': label_proportion.to_dict(),
        'std_length_words': std_length_words,
        'median_length_words': median_length_words,
        'quantiles_words': quantiles_words.to_dict(),
        'most_common_words': most_common_words,
    }

    # Save plots to files
    plots_dir = f"datasets/plots/{filename_no_ext}"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Plot the distribution of sentence lengths (words)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['sentence_length_words'], bins=50, kde=True)
    plt.title(f'Sentence Length Distribution (Words) in {filename_no_ext}')
    plt.xlabel('Sentence Length (Number of Words)')
    plt.ylabel('Frequency')
    plot_path = os.path.join(plots_dir, f'sentence_length_words_{filename_no_ext}.png')
    plt.savefig(plot_path)
    plt.close()
    report_data[filename_no_ext]['sentence_length_words_plot'] = plot_path

    # Plot the distribution of sentence lengths (chars)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['sentence_length_chars'], bins=50, kde=True, color='green')
    plt.title(f'Sentence Length Distribution (Characters) in {filename_no_ext}')
    plt.xlabel('Sentence Length (Number of Characters)')
    plt.ylabel('Frequency')
    plot_path = os.path.join(plots_dir, f'sentence_length_chars_{filename_no_ext}.png')
    plt.savefig(plot_path)
    plt.close()
    report_data[filename_no_ext]['sentence_length_chars_plot'] = plot_path

    # Plot the distribution of labels
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=data, order=label_distribution.index)
    plt.title(f'Label Distribution in {filename_no_ext}')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plot_path = os.path.join(plots_dir, f'label_distribution_{filename_no_ext}.png')
    plt.savefig(plot_path)
    plt.close()
    report_data[filename_no_ext]['label_distribution_plot'] = plot_path

    # Sentence length statistics per label
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label', y='sentence_length_words', data=data)
    plt.title(f'Sentence Length per Label in {filename_no_ext}')
    plt.xlabel('Labels')
    plt.ylabel('Sentence Length (Number of Words)')
    plot_path = os.path.join(plots_dir, f'sentence_length_per_label_{filename_no_ext}.png')
    plt.savefig(plot_path)
    plt.close()
    report_data[filename_no_ext]['sentence_length_per_label_plot'] = plot_path

    # Most common words
    common_words, common_counts = zip(*most_common_words)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(common_words), y=list(common_counts))
    plt.title(f'Most Common Words (Excluding Stop Words) in {filename_no_ext}')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plot_path = os.path.join(plots_dir, f'most_common_words_{filename_no_ext}.png')
    plt.savefig(plot_path)
    plt.close()
    report_data[filename_no_ext]['most_common_words_plot'] = plot_path

    # N-gram analysis (bigrams)
    bigrams = list(ngrams(words_no_stop, 2))
    most_common_bigrams = Counter(bigrams).most_common(20)
    bigram_phrases = [' '.join(bigram) for bigram, count in most_common_bigrams]
    bigram_counts = [count for bigram, count in most_common_bigrams]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=bigram_phrases, y=bigram_counts)
    plt.title(f'Most Common Bigrams in {filename_no_ext}')
    plt.xlabel('Bigrams')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plot_path = os.path.join(plots_dir, f'most_common_bigrams_{filename_no_ext}.png')
    plt.savefig(plot_path)
    plt.close()
    report_data[filename_no_ext]['most_common_bigrams_plot'] = plot_path

    # Stop words proportion
    total_words = len(all_words)
    num_stop_words = len(all_words) - len(words_no_stop)
    stop_words_proportion = (num_stop_words / total_words) * 100
    report_data[filename_no_ext]['stop_words_proportion'] = stop_words_proportion

def generate_report(report_data, output_filename=f"Dataset_Analysis_Report"):
    # Prepare LaTeX template
    template_str = r'''
    \documentclass{article}
    \usepackage{graphicx}
    \usepackage{float}
    \usepackage{booktabs}
    \usepackage{longtable}
    \usepackage{geometry}
    \geometry{margin=1in}
    \begin{document}
    \title{Dataset Dimensional Analysis Report}
    \author{Generated by Python Script}
    \date{\today}
    \maketitle
    {% for dataset, data in report_data.items() %}
    \section*{Analysis of {{ dataset|latex_escape }}}
    \subsection*{Basic Information}
    \begin{itemize}
        \item Number of samples: {{ data['num_samples'] }}
        \item Missing 'text' entries: {{ data['missing_text'] }}
        \item Missing 'label' entries: {{ data['missing_label'] }}
        \item Number of unique sentences: {{ data['num_unique_sentences'] }}
        \item Number of unique labels: {{ data['num_unique_labels'] }}
        \item Vocabulary size: {{ data['vocab_size'] }}
    \end{itemize}
    \subsection*{Sentence Length (Words)}
    \begin{itemize}
        \item Average: {{ data['avg_length_words']|round(2) }}
        \item Standard deviation: {{ data['std_length_words']|round(2) }}
        \item Median: {{ data['median_length_words'] }}
        \item Max: {{ data['max_length_words'] }}
        \item Min: {{ data['min_length_words'] }}
        \item Quantiles (25\%, 50\%, 75\%): {{ data['quantiles_words'][0.25] }}, {{ data['quantiles_words'][0.5] }}, {{ data['quantiles_words'][0.75] }}
    \end{itemize}
    \subsection*{Stop Words Proportion}
    {{ data['stop_words_proportion']|round(2) }}\%
    \subsection*{Label Distribution}
    \begin{tabular}{ll}
    \toprule
    Label & Frequency \\
    \midrule
    {% for label, freq in data['label_distribution'].items() %}
    {{ label|latex_escape }} & {{ freq }} \\
    {% endfor %}
    \bottomrule
    \end{tabular}
    \newline
    \newline
    \includegraphics[width=\linewidth]{ {{ data['label_distribution_plot'] }} }
    \subsection*{Sentence Length Distribution (Words)}
    \includegraphics[width=\linewidth]{ {{ data['sentence_length_words_plot'] }} }
    \subsection*{Sentence Length Distribution (Characters)}
    \includegraphics[width=\linewidth]{ {{ data['sentence_length_chars_plot'] }} }
    \subsection*{Sentence Length per Label}
    \includegraphics[width=\linewidth]{ {{ data['sentence_length_per_label_plot'] }} }
    \subsection*{Most Common Words (Excluding Stop Words)}
    \includegraphics[width=\linewidth]{ {{ data['most_common_words_plot'] }} }
    \subsection*{Most Common Bigrams}
    \includegraphics[width=\linewidth]{ {{ data['most_common_bigrams_plot'] }} }
    {% endfor %}
    \end{document}
    '''

    # Define the LaTeX escaping function
    def latex_escape(s):
        """
        Escapes LaTeX special characters in a string.
        """
        s = s.replace('\\', r'\textbackslash{}')
        s = s.replace('&', r'\&')
        s = s.replace('%', r'\%')
        s = s.replace('$', r'\$')
        s = s.replace('#', r'\#')
        s = s.replace('_', r'\_')
        s = s.replace('{', r'\{')
        s = s.replace('}', r'\}')
        s = s.replace('~', r'\textasciitilde{}')
        s = s.replace('^', r'\textasciicircum{}')
        return s

    # Create a Jinja2 environment and register the filter
    env = Environment()
    env.filters['latex_escape'] = latex_escape

    # Render the template
    template = env.from_string(template_str)
    rendered_latex = template.render(report_data=report_data)

    # Save the LaTeX file
    with open(f'{output_filename}.tex', 'w') as f:
        f.write(rendered_latex)

    # Compile the LaTeX file to PDF
    try:
        subprocess.run(['pdflatex', f'{output_filename}.tex'], check=True)
    except FileNotFoundError:
        print(
            "Error: 'pdflatex' command not found. Please ensure that LaTeX is installed and 'pdflatex' is available in your system PATH.")
    except subprocess.CalledProcessError as e:
        print("Error during LaTeX compilation:", e)
        print("Please check the LaTeX log for more details.")
    finally:
        # Clean up auxiliary files generated by LaTeX
        aux_files = ['.aux', '.log', '.out']
        for ext in aux_files:
            file = f'{output_filename}{ext}'
            if os.path.exists(file):
                os.remove(file)

# Main execution
if __name__ == '__main__':
    report_data = {}
    analyze_dataset('datasets/reduced_emotion_test.csv', report_data)
    analyze_dataset('datasets/reduced_emotion_train.csv', report_data)
    generate_report(report_data, output_filename="Dataset_Analysis_Report_Emotion_Reduced")
