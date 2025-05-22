import { css } from "@emotion/react"
import { useEffect, useState } from "react"
import { useParams } from "react-router"
import { Div, H1 } from "style-props-html"
import contentCatalog from "./assets/content-catalog.json"

type TopicData = Array<{
    topic: string,
    summary: string,
    paragraphs: Array<string>
}>


function MainContent(
    {
        data
    }: {
        data: TopicData
    }
) {
    const [selectedTopic, setSelectedTopic] = useState<TopicData[number] | null>(null)
    return <Div display="grid" width="100dvw" height="100dvh" overflow="hidden" gridTemplateRows="1fr" gridTemplateColumns="20vw 1fr 1fr">
        <Div borderRight="2px solid black" height="100dvh" display="flex" flexDirection="column" overflowX="hidden" overflowY="auto" alignItems="flex-start" justifyContent="flex-start">
            {
                data.map(({ topic }, i) => {
                    return <Div key={i} position="relative" width="100%" background="hsl(0, 0%, 80%)" onClick={() => {
                        setSelectedTopic(data[i])
                    }}>

                        <Div padding="6px" paddingRight="18px" fontSize="16px" position="relative" visibility="hidden" pointerEvents="none" width="100%">{topic}</Div>
                        <Div padding="6px" paddingRight="18px" fontSize="16px" position="absolute" top="0" left="0" right="0" bottom="0" visibility="visible" pointerEvents="auto" transformOrigin="top left" cursor="pointer" userSelect="none" color="hsl(240,80%,50%)" css={css`
                            z-index: 1;
                            text-decoration: none;    
                            &:hover {
                                z-index: 2;
                                transform: scale(1.05);
                                text-decoration: underline;
                            }
                            &:active {
                                transform: scale(0.95);
                                text-decoration: underline;
                            }

                            `}>{topic}</Div>
                    </Div>
                })
            }
        </Div>

        <Div borderRight="2px solid black" display="grid" height="100dvh" gridTemplateColumns="1fr" gridTemplateRows="auto 1fr">
            <Div borderBottom="1px">
                <H1 width="100%" textAlign="left" fontSize="18">
                    {selectedTopic ? selectedTopic.topic : "No topic selected"}
                </H1>
            </Div>
            <Div overflowY="auto" whiteSpace="pre-wrap">
                {
                    selectedTopic?.summary
                }
            </Div>
        </Div>

        <Div display="grid" height="100dvh" gridTemplateColumns="1fr" gridTemplateRows="auto 1fr">
            <Div borderBottom="1px">
                <H1 width="100%" textAlign="left" fontSize="18">
                    {selectedTopic ? "Original Text" : "No topic selected"}
                </H1>
            </Div>
            <Div overflowY="auto">

                <Div whiteSpace="pre-wrap">

                    {
                        selectedTopic?.paragraphs.join("\n\n")
                    }
                </Div>

            </Div>
        </Div>
    </Div>
}

export default function Viewer() {
    const { jsonFile } = useParams()
    const [data, setData] = useState<TopicData | null>(null)
    async function fetchData() {
        const response = await fetch(contentCatalog[jsonFile as keyof typeof contentCatalog])
        const json = await response.json()
        setData(json)
    }
    useEffect(() => {
        fetchData()
    }, [jsonFile])

    return <>{data && <MainContent data={data} />}</>
}