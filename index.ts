import { OpenAI } from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const sentences = [
    'This is a test.',
    'It is Thursday, the 31st of August.',
    'I am an AI language model created by OpenAI.',
    'Testing embeddings!',
    'Built with VSCode.',
    'NodeJS is written in C++.',
    'g++ is written in C++ too.',
    'rustc is written in Rust.',
    'import OpenAI from "openai";',
];

const query = 'Programming languages and IDEs.';

async function generateEmbeddings(input: string): Promise<number[]> {
    return (await openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input
    })).data[0].embedding;
}

async function main() {
    const queryEmbeddings = await generateEmbeddings(query);
    const sentenceEmbeddings = await Promise.all(sentences.map(generateEmbeddings));

    const rankedSentences = sentenceEmbeddings.map((sentenceEmbedding, index) => {
        const score = queryEmbeddings.reduce((acc, curr, i) => acc + (curr - sentenceEmbedding[i]) ** 2, 0);
        return { sentence: sentences[index], score };
    }
    ).sort((a, b) => b.score - a.score);

    console.log(rankedSentences);
}
main();
