import * as spacy from 'spacy';
import { TextBlob } from 'textblob';
import { mean } from 'lodash';
import { cosineSimilarity } from 'ml-distance';
import { parseISO } from 'date-fns';

const nlp = spacy.load('en_core_web_sm');

function getActionVector(action) {
    const doc = nlp(action);
    if (doc.vector_norm) {
        return doc.vector;
    } else {
        // fallback: average token vectors
        const vectors = doc.tokens.map(token => token.has_vector ? token.vector : null).filter(v => v);
        if (vectors.length) {
            return mean(vectors);
        } else {
            return new Array(nlp.vocab.vectors_length).fill(0);
        }
    }
}

function actionSimilarity(action1, action2) {
    const v1 = getActionVector(action1);
    const v2 = getActionVector(action2);
    if (v1.length === 0 || v2.length === 0) {
        return 0.0;
    }
    return cosineSimilarity([v1], [v2])[0][0];
}

class Predictor {
    constructor(model, allKeywords, actions) {
        this.model = model;
        this.allKeywords = allKeywords;
        this.actions = actions;
    }

    extractAction(text) {
        const doc = nlp(text);
        // Try to find the main verb or action
        for (const token of doc.tokens) {
            if (token.pos === 'VERB' && !token.is_stop) {
                return token.lemma.toLowerCase();
            }
        }
        // fallback: first noun
        for (const token of doc.tokens) {
            if (token.pos === 'NOUN' && !token.is_stop) {
                return token.lemma.toLowerCase();
            }
        }
        return '';
    }

    logToVector(keywords, polarity, subjectivity, action, userTime) {
        const kws = new Set(keywords.split(',').map(k => k.trim()).filter(k => k));
        const keywordVec = this.allKeywords.map(k => kws.has(k) ? 1 : 0);
        // Action embedding (spaCy vector, reduced to mean for simplicity)
        const doc = action ? nlp(action) : null;
        let actionVec;
        if (doc && doc.vector_norm) {
            actionVec = doc.vector.slice(0, 10).concat(new Array(10 - doc.vector.length).fill(0));
        } else {
            actionVec = new Array(10).fill(0);
        }
        // Time features: extract hour and weekday from ISO string
        let hour = 0.0;
        let weekday = 0.0;
        try {
            const dt = parseISO(userTime);
            hour = dt.getHours() / 23.0;  // normalize
            weekday = dt.getDay() / 6.0;  // normalize (0=Mon, 6=Sun)
        } catch (error) {
            hour = 0.0;
            weekday = 0.0;
        }
        return [...keywordVec, polarity, subjectivity, ...actionVec, hour, weekday];
    }

    extractFeaturesFromText(text) {
        const doc = nlp(text);
        const keywords = new Set();
        for (const ent of doc.ents) {
            keywords.add(ent.text.toLowerCase());
        }
        for (const token of doc.tokens) {
            if (['NOUN', 'ADJ', 'VERB'].includes(token.pos) && !token.is_stop) {
                keywords.add(token.lemma.toLowerCase());
            }
        }
        const blob = new TextBlob(text);
        const polarity = blob.sentiment.polarity;
        const subjectivity = blob.sentiment.subjectivity;
        const action = this.extractAction(text);
        return [Array.from(keywords).join(', '), polarity, subjectivity, action];
    }

    predict(query, predTime) {
        const [keywords, polarity, subjectivity, action] = this.extractFeaturesFromText(query);
        const xPred = this.logToVector(keywords, polarity, subjectivity, action, predTime);
        const prob = this.model.predict([xPred])[0][0];
        return [Math.round(prob * 100), action];
    }

    prepareTrainingData(data) {
        const X = [];
        const y = [];
        for (const row of data) {
            const [userInput, keywords, polarity, subjectivity, targetAction, userTime] = row;
            const action = targetAction || this.extractAction(userInput);
            X.push(this.logToVector(keywords, polarity, subjectivity, action, userTime));
            y.push(action.includes(userInput.toLowerCase()) || keywords.includes(action) ? 1 : 0);
        }
        return [X, y];
    }
}

