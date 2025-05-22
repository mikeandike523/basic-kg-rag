import { A, Div, H1, Ul } from 'style-props-html'

import contentCatalog from './assets/content-catalog.json'

const items = Object.keys(contentCatalog)

export default function Home() {
    return <Div>
        <H1>Topicization Examples</H1>
        <Ul>
            {items.map((item) => {
                return <A key={item} href={`/#/viewer/${item}`}>Mistral 7B (Quantized, 8-bit)</A>
            })}
        </Ul>
    </Div>
}