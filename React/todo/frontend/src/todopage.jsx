import React, { Component } from "react"; 
import Header from "./header";
import Form from "./form";

export default class Todopage extends Component {



render(){

    return(<React.Fragment>
        <div className='container'>
            <div className='app-wrapper'>
                <div>
                    <Header/>
                </div>
                <div>
                    <Form></Form>
                </div>
            </div>

        </div>
        
        </React.Fragment>
    )


}



} // Component Parenthesis
